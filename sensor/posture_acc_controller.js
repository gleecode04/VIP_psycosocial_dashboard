import { createDriver, SERVICES } from './driver.js';

// Training data is ~10 readings/second (see posture_ml/windows.py:
// WINDOW_SIZE_READINGS=10 readings per 1s window), but the Polar H10
// streams ACC at 100Hz. Raw samples are averaged into 100ms buckets before
// being reported as a "reading", so a live window has the same temporal
// resolution the model was trained on instead of 10x finer, noisier data.
//
// Readings carry raw {ax, ay, az} in g, not a precomputed angle: a
// single-axis pitch formula bakes in an assumption about how the sensor is
// mounted (e.g. flat against the chest), which breaks when it's strapped
// somewhere else, like the lower back (see posture_dbr.py's module
// docstring). Sending the raw axes lets the server calibrate a neutral
// baseline per session and compute a mount-orientation-agnostic tilt angle
// instead.
const READING_BUCKET_MS = 100;
const ACC_RANGE_G = 4;
const ACC_RATE_HZ = 100;
const ACC_RESOLUTION_BITS = 16;

// Mirrors sensor/controller.js's connect/retry/disconnect lifecycle; kept as
// a separate class (rather than branching PolarController) because it drives
// a different PMD stream (ACC, not ECG) with a different payload shape.
export class PostureAccController {
  constructor({ bluetooth = globalThis.navigator?.bluetooth, makeDriver = createDriver,
    onChange = () => {}, onReading = () => {}, retryBaseMs = 1000,
    maxRetries = 5, timeoutMs = 20000 } = {}) {
    Object.assign(this, { bluetooth, makeDriver, onChange, onReading, retryBaseMs, maxRetries, timeoutMs });
    this.state = 'idle';
    this.message = 'No Polar H10 connected.';
    this.generation = 0;
    this.attempt = 0;
    this.wanted = false;
    this.busy = false;
    this._sessionStartMs = null;
    this._bucketIndex = null;
    this._bucketSumAx = 0;
    this._bucketSumAy = 0;
    this._bucketSumAz = 0;
    this._bucketCount = 0;
    this.lost = () => {
      if (!this.wanted || this.busy) return;
      this.release();
      this.scheduleRetry('Sensor disconnected.');
    };
  }

  status(state, message) {
    console.log(`[posture-ble] status -> ${state}: ${message}`);
    this.state = state;
    this.message = message;
    this.onChange(this);
  }

  async connect() {
    if (this.busy || this.wanted) return;
    if (!this.bluetooth) return this.status('error', 'Web Bluetooth requires Chrome or Edge on a supported platform, using localhost or HTTPS.');
    this.busy = true;
    const generation = ++this.generation;
    this.status('connecting', 'Select a Polar H10 in the Bluetooth chooser.');
    try {
      const device = await this.bluetooth.requestDevice({ filters: [{ namePrefix: 'Polar H10' }], optionalServices: SERVICES });
      console.log(`[posture-ble] requestDevice resolved: name=${device.name} id=${device.id}`);
      if (generation !== this.generation) return;
      this.device = device;
      this.device.addEventListener('gattserverdisconnected', this.lost);
      this.wanted = true;
      this.attempt = 0;
    } catch (error) {
      if (generation === this.generation) this.status('error', error.name === 'NotFoundError' ? 'No device selected. Connect to try again.' : `Bluetooth selection failed: ${error.message}`);
    } finally {
      this.busy = false;
    }
    if (generation === this.generation && this.wanted) await this.open(generation);
  }

  async open(generation) {
    if (this.busy || !this.wanted || generation !== this.generation) return;
    this.busy = true;
    this.status(this.attempt ? 'reconnecting' : 'connecting', 'Initializing Polar H10…');
    let failure;
    let timer;
    const scope = this.makeDriver(this.device, () => {
      this.packetErrors = (this.packetErrors || 0) + 1;
    }, true); // verbose: surface the library's own step-by-step log (fast-path vs secured-pairing fallback) in the console, to see exactly where a stuck connection is stalling
    this.scope = scope;
    const current = () => this.wanted && generation === this.generation && this.scope === scope;
    const check = () => {
      if (!current() || !this.device.gatt?.connected) throw new Error('Sensor disconnected or connection cancelled');
    };
    try {
      const setup = async () => {
        const p = scope.driver;
        p.addEventListener('ACC', (frame) => {
          if (current()) this._handleAccFrame(frame);
        });
        console.log('[posture-ble] init() starting');
        const t0 = performance.now();
        await p.init(100);
        console.log(`[posture-ble] init() done in ${(performance.now() - t0).toFixed(0)}ms`);
        check();
        // A brief settle pause before commanding a stream start: the
        // reference polar-h10-visualizer app has the user click a SEPARATE
        // button to start a stream well after connecting, which naturally
        // gives the device a moment after init/bonding before the next PMD
        // control-point write. Our flow does both back-to-back
        // programmatically with no such gap -- this restores one.
        await new Promise((resolve) => setTimeout(resolve, 400));
        check();
        console.log('[posture-ble] startACC() starting');
        const t1 = performance.now();
        const reply = await p.startACC(ACC_RANGE_G, ACC_RATE_HZ, ACC_RESOLUTION_BITS);
        console.log(`[posture-ble] startACC() done in ${(performance.now() - t1).toFixed(0)}ms, reply=`, reply);
        if (!['SUCCESS', 'ALREADY IN STATE'].includes(reply?.error))
          throw new Error(reply?.error || 'No ACC start acknowledgement');
        check();
      };
      await Promise.race([setup(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Connection timed out')), this.timeoutMs);
      })]);
      if (current()) {
        this._sessionStartMs = null;
        this._bucketIndex = null;
        this.status('connected', `Connected to ${this.device.name || 'Polar H10'}. Streaming accelerometer data.`);
      }
    } catch (error) {
      failure = error;
      if (current()) {
        this.release();
        this.device.gatt?.disconnect();
      }
    } finally {
      clearTimeout(timer);
      // Only clear busy for the attempt that's still current -- a stale,
      // already-abandoned attempt (superseded by a newer connect() or a
      // disconnect() that already reset busy itself) must not clobber a
      // different, newer attempt's busy=true.
      if (generation === this.generation) this.busy = false;
    }
    if (failure && generation === this.generation && this.wanted) this.scheduleRetry(failure.message);
  }

  _handleAccFrame(frame) {
    const samples = frame.samples;
    const times = frame.epoch_timestamps_ms;
    if (!samples || !times) return;
    const n = Math.floor(samples.length / 3);
    for (let i = 0; i < n; i++) {
      const ax = samples[i * 3] / 1000;
      const ay = samples[i * 3 + 1] / 1000;
      const az = samples[i * 3 + 2] / 1000;
      if (this._sessionStartMs === null) this._sessionStartMs = times[i];
      this._accumulate(times[i] - this._sessionStartMs, ax, ay, az);
    }
  }

  _accumulate(elapsedMs, ax, ay, az) {
    const bucket = Math.floor(elapsedMs / READING_BUCKET_MS);
    if (this._bucketIndex === null) this._bucketIndex = bucket;
    if (bucket !== this._bucketIndex) {
      if (this._bucketCount > 0) {
        this.onReading({
          t: (this._bucketIndex * READING_BUCKET_MS) / 1000,
          ax: this._bucketSumAx / this._bucketCount,
          ay: this._bucketSumAy / this._bucketCount,
          az: this._bucketSumAz / this._bucketCount,
        });
      }
      this._bucketIndex = bucket;
      this._bucketSumAx = 0;
      this._bucketSumAy = 0;
      this._bucketSumAz = 0;
      this._bucketCount = 0;
    }
    this._bucketSumAx += ax;
    this._bucketSumAy += ay;
    this._bucketSumAz += az;
    this._bucketCount += 1;
  }

  scheduleRetry(reason) {
    if (this.retryTimer || !this.wanted) return;
    if (this.attempt >= this.maxRetries) {
      this.disconnect();
      this.status('error', `${reason} Reconnection attempts exhausted. Check battery/range and OS pairing (firmware 4+), then Connect again.`);
      return;
    }
    const delay = Math.min(this.retryBaseMs * 2 ** this.attempt, 16000);
    this.attempt++;
    this.status('reconnecting', `${reason} Retrying ${this.attempt}/${this.maxRetries} in ${(delay / 1000).toFixed(1)}s.`);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      void this.open(this.generation);
    }, delay);
  }

  release() {
    this.scope?.dispose();
    this.scope = null;
  }

  disconnect() {
    this.generation++;
    this.wanted = false;
    // Force this now, don't wait for a stale in-flight open() call's own
    // finally block to get around to it (up to timeoutMs later, and even
    // then without ever re-rendering) -- otherwise clicking Disconnect
    // mid-attempt leaves Connect stuck disabled until that abandoned
    // attempt eventually finishes on its own.
    this.busy = false;
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
    this.release();
    this.device?.removeEventListener('gattserverdisconnected', this.lost);
    this.device?.gatt?.disconnect();
    this.status('idle', 'No Polar H10 connected.');
  }
}
