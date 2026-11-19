// node_modules/polar-h10/dist/esm/index.js
var PolarSettingType = /* @__PURE__ */ ((PolarSettingType2) => {
  PolarSettingType2[PolarSettingType2["SAMPLE_RATE"] = 0] = "SAMPLE_RATE";
  PolarSettingType2[PolarSettingType2["RESOLUTION"] = 1] = "RESOLUTION";
  PolarSettingType2[PolarSettingType2["RANGE_PN_UNIT"] = 2] = "RANGE_PN_UNIT";
  PolarSettingType2[PolarSettingType2["RANGE_MILI_UNIT"] = 3] = "RANGE_MILI_UNIT";
  PolarSettingType2[PolarSettingType2["NUM_CHANNELS"] = 4] = "NUM_CHANNELS";
  PolarSettingType2[PolarSettingType2["CONVERSION_FACTOR"] = 5] = "CONVERSION_FACTOR";
  return PolarSettingType2;
})(PolarSettingType || {});
function parseUint16(d, offset, little_endian = true) {
  return d.getUint16(offset, little_endian);
}
function parseFloat32(d, offset, little_endian = true) {
  return d.getFloat32(offset, little_endian);
}
function parse4xUint16(d, offset, little_endian = true) {
  return [
    d.getUint16(offset, little_endian),
    d.getUint16(offset + 2, little_endian),
    d.getUint16(offset + 4, little_endian),
    d.getUint16(offset + 6, little_endian)
  ];
}
var setting_parsers = {
  SAMPLE_RATE: parseUint16,
  RESOLUTION: parseUint16,
  RANGE_PN_UNIT: parseUint16,
  RANGE_MILI_UNIT: parse4xUint16,
  CONVERSION_FACTOR: parseFloat32
};
var setting_parser_offsets = {
  SAMPLE_RATE: 2,
  RESOLUTION: 2,
  RANGE_PN_UNIT: 2,
  RANGE_MILI_UNIT: 8,
  NUM_CHANNELS: 1,
  CONVERSION_FACTOR: 4
};
var PolarPMDCommand = /* @__PURE__ */ ((PolarPMDCommand2) => {
  PolarPMDCommand2[PolarPMDCommand2["GET_MEASUREMENT_SETTINGS"] = 1] = "GET_MEASUREMENT_SETTINGS";
  PolarPMDCommand2[PolarPMDCommand2["REQUEST_MEASUREMENT_START"] = 2] = "REQUEST_MEASUREMENT_START";
  PolarPMDCommand2[PolarPMDCommand2["REQUEST_MEASUREMENT_STOP"] = 3] = "REQUEST_MEASUREMENT_STOP";
  return PolarPMDCommand2;
})(PolarPMDCommand || {});
var ERROR_MSGS = [
  "SUCCESS",
  "INVALID OP CODE",
  "INVALID MEASUREMENT TYPE",
  "NOT SUPPORTED",
  "INVALID LENGTH",
  "INVALID PARAMETER",
  "ALREADY IN STATE",
  "INVALID RESOLUTION",
  "INVALID SAMPLE RATE",
  "INVALID RANGE",
  "INVALID MTU",
  "INVALID NUMBER OF CHANNELS",
  "INVALID STATE",
  "DEVICE IN CHARGER"
];
var PMD_SERVICE_ID = "fb005c80-02e7-f387-1cad-8acd2d8df0c8";
var PMD_CTRL_CHAR = "fb005c81-02e7-f387-1cad-8acd2d8df0c8";
var PMD_DATA_CHAR = "fb005c82-02e7-f387-1cad-8acd2d8df0c8";
var HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb";
var HEART_RATE_MEASUREMENT_CHARACTERISTIC_UUID = "00002a37-0000-1000-8000-00805f9b34fb";
var SERVICES = [
  PMD_SERVICE_ID,
  HEART_RATE_SERVICE_UUID,
  "battery_service"
];
var PolarSensorType = /* @__PURE__ */ ((PolarSensorType2) => {
  PolarSensorType2[PolarSensorType2["ECG"] = 0] = "ECG";
  PolarSensorType2[PolarSensorType2["PPG"] = 1] = "PPG";
  PolarSensorType2[PolarSensorType2["ACC"] = 2] = "ACC";
  PolarSensorType2[PolarSensorType2["PPI"] = 3] = "PPI";
  PolarSensorType2[PolarSensorType2["GYRO"] = 5] = "GYRO";
  PolarSensorType2[PolarSensorType2["MAGNETOMETER"] = 6] = "MAGNETOMETER";
  PolarSensorType2[PolarSensorType2["SDK_MODE"] = 9] = "SDK_MODE";
  PolarSensorType2[PolarSensorType2["LOCATION"] = 10] = "LOCATION";
  PolarSensorType2[PolarSensorType2["PRESSURE"] = 11] = "PRESSURE";
  PolarSensorType2[PolarSensorType2["TEMPERATURE"] = 12] = "TEMPERATURE";
  return PolarSensorType2;
})(PolarSensorType || {});
var PolarSettingNames = Object.keys(PolarSettingType).filter(
  (t) => isNaN(Number(t))
);
var PolarSensorNames = Object.keys(PolarSensorType).filter(
  (t) => isNaN(Number(t))
);
var PolarPMDCommandNames = Object.keys(PolarPMDCommand).filter(
  (t) => isNaN(Number(t))
);
function readBits(buffer, bitOffset, bitLength) {
  let value = 0;
  for (let i = 0; i < bitLength; i++) {
    const totalBitOffset = bitOffset + i;
    const byteIndex = Math.floor(totalBitOffset / 8);
    const bitIndex = totalBitOffset % 8;
    if (byteIndex >= buffer.length) break;
    const bit = buffer[byteIndex] >> bitIndex & 1;
    value |= bit << i;
  }
  const signBit = 1 << bitLength - 1;
  if (value & signBit) {
    value |= ~((1 << bitLength) - 1);
  }
  return value;
}
var PolarH10 = class {
  device;
  server = void 0;
  PMDService = void 0;
  PMDCtrlChar = void 0;
  PMDDataChar = void 0;
  BattService = void 0;
  BattLvlChar = void 0;
  HeartRateService = void 0;
  HeartRateChar = void 0;
  streaming = false;
  verbose = true;
  dataHandleDict = {};
  heartRateHandleList = [];
  battLevelHandleList = [];
  timeOffset = BigInt(0);
  eventTimeOffset;
  lastECGTimestamp;
  lastACCTimestamp;
  ACCStarted = false;
  ECGStarted = false;
  battLvl;
  constructor(device, verbose = true) {
    this.device = device;
    this.verbose = verbose;
    this.lastECGTimestamp = 0;
    this.lastACCTimestamp = 0;
    this.ACCStarted = false;
    this.ECGStarted = false;
    for (let i = 0; i < PolarSensorNames.length; i++) {
      this.dataHandleDict[PolarSensorNames[i]] = [];
    }
    this.eventTimeOffset = 0;
    this.battLvl = 0;
  }
  addEventListener(type, handler) {
    if (!this.dataHandleDict[type].includes(handler)) {
      this.dataHandleDict[type].push(handler);
    }
  }
  addHeartRateEventListener(handler) {
    if (!this.heartRateHandleList.includes(handler)) {
      this.heartRateHandleList.push(handler);
    }
  }
  addBatteryLevelEventListener(handler) {
    if (!this.battLevelHandleList.includes(handler)) {
      this.battLevelHandleList.push(handler);
    }
  }
  removeEventListener(type, handler) {
    let index = this.dataHandleDict[type].indexOf(handler);
    if (index > -1) {
      this.dataHandleDict[type].splice(index, 1);
    }
    return index;
  }
  removeHeartRateEventListener(handler) {
    let index = this.heartRateHandleList.indexOf(handler);
    if (index > -1) {
      this.heartRateHandleList.splice(index, 1);
    }
    return index;
  }
  clearEventListner(type) {
    delete this.dataHandleDict[type];
    this.dataHandleDict[type] = [];
  }
  log(...o) {
    if (this.verbose) {
      console.log(...o);
    }
  }
  // Enhanced defensive notification handler
  async safeStartNotifications(characteristic, label) {
    if (!characteristic) return;
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        await new Promise((resolve) => setTimeout(resolve, 100));
        await characteristic.startNotifications();
        this.log(`    Successfully subscribed to ${label} notifications`);
        return;
      } catch (error) {
        this.log(
          `    \u26A0\uFE0F Attempt ${attempt} failed for ${label}: ${error.message || error}`
        );
        if (error.message.includes("range") || error.message.includes("disconnect")) {
          throw error;
        }
        if (attempt === 2) {
          throw error;
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }
  }
  resetGATTState() {
    this.PMDService = void 0;
    this.PMDCtrlChar = void 0;
    this.PMDDataChar = void 0;
    this.BattService = void 0;
    this.BattLvlChar = void 0;
    this.HeartRateService = void 0;
    this.HeartRateChar = void 0;
    this.streaming = false;
    this.ACCStarted = false;
    this.ECGStarted = false;
  }
  async writePMDCtrl(cmd_buf) {
    if (!this.PMDCtrlChar) {
      throw new Error("PMD Control Characteristic unavailable.");
    }
    const bufSource = cmd_buf.buffer;
    try {
      if (typeof this.PMDCtrlChar.writeValueWithResponse === "function") {
        await this.PMDCtrlChar.writeValueWithResponse(bufSource);
      } else {
        await this.PMDCtrlChar.writeValue(bufSource);
      }
    } catch (err) {
      this.log(
        `    (writeValue failed: ${err.message || err}, falling back to writeValueWithoutResponse...)`
      );
      await this.PMDCtrlChar.writeValueWithoutResponse(bufSource);
    }
  }
  async awaitWithTimeout(promise, timeoutMs = 5e3, label = "PMD Control command") {
    let timeoutId;
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(new Error(`${label} timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
    });
    try {
      const result = await Promise.race([promise, timeoutPromise]);
      clearTimeout(timeoutId);
      return result;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }
  async cacheGATTTable() {
    if (!this.server || !this.server.connected) {
      throw new Error("GATT Server not connected.");
    }
    this.log(`  \u{1F50D} Caching GATT Table...`);
    this.PMDService = await this.server.getPrimaryService(PMD_SERVICE_ID);
    this.PMDCtrlChar = await this.PMDService.getCharacteristic(PMD_CTRL_CHAR);
    this.PMDDataChar = await this.PMDService.getCharacteristic(PMD_DATA_CHAR);
    this.BattService = await this.server.getPrimaryService("battery_service");
    this.BattLvlChar = await this.BattService.getCharacteristic("battery_level");
    this.HeartRateService = await this.server.getPrimaryService(
      HEART_RATE_SERVICE_UUID
    );
    this.HeartRateChar = await this.HeartRateService.getCharacteristic(
      HEART_RATE_MEASUREMENT_CHARACTERISTIC_UUID
    );
    this.log(`  [Success] GATT Table cached.`);
  }
  async setupSubscriptions() {
    await this.safeStartNotifications(this.PMDCtrlChar, "PMD Control");
    this.PMDDataChar?.addEventListener(
      "characteristicvaluechanged",
      this.PMDDataHandle.bind(this)
    );
    await this.safeStartNotifications(this.PMDDataChar, "PMD Data");
    this.BattLvlChar?.addEventListener(
      "characteristicvaluechanged",
      this.BatteryLevelHandle.bind(this)
    );
    await this.safeStartNotifications(this.BattLvlChar, "Battery Level");
    this.HeartRateChar?.addEventListener(
      "characteristicvaluechanged",
      this.HearRateHandle.bind(this)
    );
    await this.safeStartNotifications(this.HeartRateChar, "Heart Rate");
  }
  // Outer initialization manager with Two-Step Connection Architecture
  async init(pairing_wait_ms = 1e4, retryAttempt = 0) {
    this.device.removeEventListener(
      "gattserverdisconnected",
      this.handleNativeDisconnect
    );
    this.device.addEventListener(
      "gattserverdisconnected",
      this.handleNativeDisconnect.bind(this)
    );
    try {
      this.log(
        `Connecting to ${this.device.name} GATT server (Step A Fast-Path, Attempt ${retryAttempt + 1})...`
      );
      this.server = await this.device.gatt?.connect();
      if (!this.server || !this.server.connected) {
        throw new Error("GATT Server failed to connect.");
      }
      await this.cacheGATTTable();
      await this.setupSubscriptions();
      await this.PMDCtrlChar?.readValue();
      this.streaming = false;
      this.log("[Success] Polar H10 Driver Initialized (Fast-path unsecured)");
      return;
    } catch (error) {
      this.log(
        `  \u26A0\uFE0F Step A unsecured initialization failed (${error.message || error}). Falling back to Step B secured flow...`
      );
      this.resetGATTState();
    }
    try {
      if (!this.server || !this.server.connected) {
        this.log(`Re-connecting GATT server for Step B...`);
        this.server = await this.device.gatt?.connect();
        if (!this.server || !this.server.connected) {
          throw new Error("GATT Server failed to re-connect in Step B.");
        }
      }
      this.log(`  Phase 1: Forcing OS Pairing Prompt via secure write...`);
      const pmdService = await this.server.getPrimaryService(PMD_SERVICE_ID);
      const pmdCtrlChar = await pmdService.getCharacteristic(PMD_CTRL_CHAR);
      try {
        const dummyCmd = new Uint8Array([1, 0]);
        await pmdCtrlChar.writeValue(dummyCmd);
      } catch (writeErr) {
        this.log(
          `    (Write triggered security sequence: ${writeErr.message || writeErr})`
        );
        await new Promise((resolve) => setTimeout(resolve, pairing_wait_ms));
      }
      this.resetGATTState();
      if (!this.server || !this.server.connected) {
        this.log(`Re-connecting GATT server post-pairing...`);
        this.server = await this.device.gatt?.connect();
      }
      this.log(`  Phase 2: Re-discovering GATT Table post-pairing...`);
      await this.cacheGATTTable();
      this.log(`  Phase 3: Starting Subscriptions...`);
      await this.setupSubscriptions();
      this.streaming = false;
      this.log("[Success] Polar H10 Driver Initialized (Secured flow)");
    } catch (error) {
      const isPairingDrop = error.message?.includes("range") || error.message?.includes("disconnected");
      if (isPairingDrop && retryAttempt < 1) {
        this.log(
          `[linking] Connection severed due to OS Pairing/Bonding cycle. Retrying...`
        );
        this.resetGATTState();
        await new Promise((resolve) => setTimeout(resolve, 1e3));
        return this.init(pairing_wait_ms, retryAttempt + 1);
      }
      if (error.message?.includes("range") && retryAttempt >= 1) {
        this.log(
          `[Fatal]: Chromium Web Bluetooth encountered a Ghost Handle state.`
        );
        this.log(
          `You MUST use OS-level Bluetooth pairing before connecting.`
        );
      } else {
        this.log(`[Failed] Initialization failed: ${error.message || error}`);
      }
      throw error;
    }
  }
  handleNativeDisconnect() {
    this.log("[!!!] GATT Server disconnected natively.");
    this.resetGATTState();
  }
  PMDCtrlCharHandle(event) {
    this.log(event);
  }
  PMDCtrlDataHandle(event) {
    this.log(event);
  }
  async getBatteryLevel() {
    let battRead = await this.BattLvlChar?.readValue();
    if (battRead) {
      return battRead.getUint8(0);
    } else {
      return 0;
    }
  }
  async getPMDFeatures() {
    const PMEFeatures = await this.PMDCtrlChar?.readValue();
    const featureList = [];
    this.log(PMEFeatures);
    if (PMEFeatures !== void 0) {
      if (PMEFeatures !== void 0 && PMEFeatures.byteLength >= 2) {
        const feature_num = PMEFeatures.getUint8(1);
        for (let i = 0; i < PolarSensorNames.length; i++) {
          const sensor_name = PolarSensorNames[i];
          if (feature_num >> PolarSensorType[sensor_name] & 1) {
            featureList.push(sensor_name);
          }
        }
      }
    }
    return featureList;
  }
  async getSensorSettingsFromName(sensorName) {
    return this.getSensorSettingsFromId(PolarSensorType[sensorName]);
  }
  parseSensorSettings(val) {
    if (val.getUint8(0) == 240 && val.getUint8(1) == 1) {
      const info = {
        type: PolarSensorType[val.getUint8(2)],
        error: ERROR_MSGS[val.getUint8(3)],
        more_frames: val.getUint8(4),
        settings: {}
      };
      let i = 5;
      while (i < val.byteLength) {
        const setting_type = val.getUint8(i);
        i += 1;
        const arr_len = val.getUint8(i);
        i += 1;
        const setting_name = PolarSettingType[setting_type];
        info.settings[setting_name] = [];
        for (let arr_i = 0; arr_i < arr_len; arr_i++) {
          info.settings[setting_name].push(
            setting_parsers[setting_name](val, i)
          );
          i += setting_parser_offsets[setting_name];
        }
      }
      return info;
    }
  }
  async getSensorSettingsFromId(sensorId) {
    if (!this.streaming) {
      let sensorSettingPromiseRSLV;
      const sensorSettingPromise = new Promise((rslv) => {
        sensorSettingPromiseRSLV = rslv;
      });
      const PMDSensorSettingHandle = (event) => {
        const val = event?.target?.value;
        sensorSettingPromiseRSLV(this.parseSensorSettings(val));
      };
      this.PMDCtrlChar?.addEventListener(
        "characteristicvaluechanged",
        PMDSensorSettingHandle,
        { once: true }
      );
      const cmd_buf = new Uint8Array([
        1,
        sensorId
      ]);
      await this.writePMDCtrl(cmd_buf);
      return await this.awaitWithTimeout(sensorSettingPromise, 5e3, "Get Settings");
    }
  }
  HearRateHandle(event) {
    if (this.heartRateHandleList.length > 0) {
      const val = event.target.value;
      const hear_rate_info = {
        heart_rate_bpm: -1,
        rr_intervals_ms: [],
        recv_epoch_time_ms: event.timeStamp + performance.timeOrigin
      };
      const flags = val.getUint8(0);
      let offset = 1;
      const heartRateFormat = flags & 1;
      if (heartRateFormat === 1) {
        hear_rate_info.heart_rate_bpm = val.getUint16(offset, true);
        offset += 2;
      } else {
        hear_rate_info.heart_rate_bpm = val.getUint8(offset);
        offset += 1;
      }
      const rrIntervalPresent = (flags & 16) !== 0;
      if (rrIntervalPresent) {
        while (offset < val.byteLength) {
          const rawRrInterval = val.getUint16(offset, true);
          const rrIntervalMs = rawRrInterval / 1024 * 1e3;
          hear_rate_info.rr_intervals_ms.push(rrIntervalMs);
          offset += 2;
        }
      }
      for (const handler of this.heartRateHandleList) {
        handler(hear_rate_info);
      }
    }
  }
  BatteryLevelHandle(event) {
    this.battLvl = event.target.value.getUint8();
    if (this.battLevelHandleList.length > 0) {
      for (const handler of this.battLevelHandleList) {
        handler(this.battLvl);
      }
    }
  }
  PMDDataHandle(event) {
    const val = event.target.value;
    const dataTimeStamp = val.getBigUint64(1, true);
    if (this.timeOffset === BigInt(0)) {
      this.timeOffset = dataTimeStamp;
      this.eventTimeOffset = event.timeStamp + performance.timeOrigin;
    }
    const offset_timestamp = Number(dataTimeStamp - this.timeOffset) / 1e6;
    const type = val.getUint8(0);
    const frame_type = val.getUint8(9);
    const dataFrame = {
      type: PolarSensorType[type],
      sample_timestamp_ms: offset_timestamp,
      prev_sample_timestamp_ms: 0,
      recv_epoch_time_ms: event.timeStamp + performance.timeOrigin,
      event_time_offset_ms: this.eventTimeOffset
    };
    let estimated_sample_interval = 0;
    let s_i_delta = 1;
    switch (type) {
      case 2:
        if (frame_type == 1) {
          dataFrame.samples = new Int16Array(val.buffer.slice(10));
          dataFrame.prev_sample_timestamp_ms = this.lastACCTimestamp;
          this.lastACCTimestamp = offset_timestamp;
          s_i_delta = 3;
        } else if (frame_type === 2) {
          const refX = val.getInt16(10, true);
          const refY = val.getInt16(12, true);
          const refZ = val.getInt16(14, true);
          const deltaSize = val.getUint8(16);
          if (deltaSize > 0) {
            const rawData = new Uint8Array(val.buffer);
            const deltaBuffer = rawData.subarray(17);
            const remainingBits = deltaBuffer.length * 8;
            const numTriplets = Math.floor(remainingBits / (deltaSize * 3));
            dataFrame.samples = new Int16Array((numTriplets + 1) * 3);
            dataFrame.samples[0] = refX;
            dataFrame.samples[1] = refY;
            dataFrame.samples[2] = refZ;
            let bitOffset = 0;
            let currentX = refX;
            let currentY = refY;
            let currentZ = refZ;
            let deltaX = 0;
            let deltaY = 0;
            let deltaZ = 0;
            for (let i = 0; i < numTriplets; i++) {
              const ddx = readBits(deltaBuffer, bitOffset, deltaSize);
              bitOffset += deltaSize;
              const ddy = readBits(deltaBuffer, bitOffset, deltaSize);
              bitOffset += deltaSize;
              const ddz = readBits(deltaBuffer, bitOffset, deltaSize);
              bitOffset += deltaSize;
              deltaX += ddx;
              deltaY += ddy;
              deltaZ += ddz;
              currentX += deltaX;
              currentY += deltaY;
              currentZ += deltaZ;
              const baseIdx = (i + 1) * 3;
              dataFrame.samples[baseIdx] = currentX;
              dataFrame.samples[baseIdx + 1] = currentY;
              dataFrame.samples[baseIdx + 2] = currentZ;
            }
          } else {
            dataFrame.samples = new Int16Array([refX, refY, refZ]);
          }
          dataFrame.prev_sample_timestamp_ms = this.lastACCTimestamp;
          this.lastACCTimestamp = offset_timestamp;
          s_i_delta = 3;
        }
        break;
      case 0:
        if (frame_type === 0) {
          const numFrames = Math.floor((val.byteLength - 10) / 3);
          dataFrame.samples = new Int32Array(numFrames);
          for (let i = 10; i < val.byteLength; i += 3) {
            let d = val.getUint8(i + 2) << 16 | val.getUint8(i + 1) << 8 | val.getUint8(i);
            if (d & 8388608) {
              d |= 4278190080;
            }
            dataFrame.samples[Math.floor((i - 10) / 3)] = d;
          }
          dataFrame.prev_sample_timestamp_ms = this.lastECGTimestamp;
          this.lastECGTimestamp = offset_timestamp;
          s_i_delta = 1;
        } else if (frame_type === 1) {
          const rawData = new Uint8Array(val.buffer);
          let refSample = val.getUint8(10) | val.getUint8(11) << 8 | val.getUint8(12) << 16;
          if (refSample & 8388608) refSample |= 4278190080;
          const deltaSize = rawData[13];
          const remainingBits = (rawData.length - 14) * 8;
          const numDeltas = Math.floor(remainingBits / deltaSize);
          dataFrame.samples = new Int32Array(numDeltas + 1);
          dataFrame.samples[0] = refSample;
          let currentSample = refSample;
          let bitOffset = 0;
          const deltaBuffer = rawData.subarray(14);
          for (let i = 0; i < numDeltas; i++) {
            const delta = readBits(deltaBuffer, bitOffset, deltaSize);
            bitOffset += deltaSize;
            currentSample += delta;
            dataFrame.samples[i + 1] = currentSample;
          }
          dataFrame.prev_sample_timestamp_ms = this.lastECGTimestamp;
          this.lastECGTimestamp = offset_timestamp;
          s_i_delta = 1;
        }
        break;
    }
    if (dataFrame.samples !== void 0 && this.dataHandleDict[PolarSensorType[type]].length > 0) {
      estimated_sample_interval = (dataFrame.sample_timestamp_ms - dataFrame.prev_sample_timestamp_ms) / (dataFrame.samples.length / s_i_delta);
      if (estimated_sample_interval > 0) {
        let timeOffset = dataFrame.event_time_offset_ms + dataFrame.prev_sample_timestamp_ms;
        const numFrame = Math.floor(dataFrame.samples.length / s_i_delta);
        dataFrame.epoch_timestamps_ms = new Float64Array(numFrame);
        for (let i = 0; i < numFrame; i++) {
          timeOffset += estimated_sample_interval;
          dataFrame.epoch_timestamps_ms[i] = timeOffset;
        }
        for (const handler of this.dataHandleDict[PolarSensorType[type]]) {
          handler(dataFrame);
        }
      }
    }
  }
  parseCtrlReply(val) {
    if (val.getUint8(0) === 240) {
      const polar_cmd = val.getUint8(1);
      if (polar_cmd === 2 || polar_cmd === 3) {
        const startReply = {
          type: PolarPMDCommand[polar_cmd],
          sensor: PolarSensorType[val.getUint8(2)],
          error: ERROR_MSGS[val.getUint8(3)],
          more_frames: val.getUint8(4)
        };
        if (val.byteLength > 5) {
          startReply.reserved = val.getUint8(5);
        }
        return startReply;
      }
    }
  }
  async startHeartRate() {
    await this.HeartRateChar?.startNotifications();
  }
  async stopheartRate() {
    await this.HeartRateChar?.stopNotifications();
  }
  async startACC(rangeG = 4, sample_rate = 100, resolution = 16) {
    if (this.ACCStarted) {
      return;
    }
    let startACCRSLV;
    const startACCPromise = new Promise(
      (rslv, rjct) => {
        startACCRSLV = rslv;
      }
    );
    const PMDSensorSettingHandle = (event) => {
      this.log("PMDSensorSettingHandle");
      const val = event?.target?.value;
      startACCRSLV(this.parseCtrlReply(val));
    };
    this.PMDCtrlChar?.addEventListener(
      "characteristicvaluechanged",
      PMDSensorSettingHandle,
      { once: true }
    );
    const cmd_buf = new Uint8Array(14);
    const cmd_buf_dataview = new DataView(cmd_buf.buffer);
    cmd_buf[0] = 2;
    cmd_buf[1] = 2;
    cmd_buf[2] = 2;
    cmd_buf[3] = 1;
    cmd_buf_dataview.setUint16(4, rangeG, true);
    cmd_buf[6] = 0;
    cmd_buf[7] = 1;
    cmd_buf_dataview.setUint16(8, sample_rate, true);
    cmd_buf[10] = 1;
    cmd_buf[11] = 1;
    cmd_buf_dataview.setUint16(12, resolution, true);
    await this.writePMDCtrl(cmd_buf);
    const startReply = await this.awaitWithTimeout(
      startACCPromise,
      5e3,
      "Start ACC"
    );
    if (startReply?.error === ERROR_MSGS[0]) {
      this.ACCStarted = true;
    }
    return startReply;
  }
  async startECG(sample_rate = 130, resolution = 14) {
    if (this.ECGStarted) {
      return;
    }
    let startECGRSLV;
    const startECGPromise = new Promise(
      (rslv) => {
        startECGRSLV = rslv;
      }
    );
    const PMDSensorSettingHandle = (event) => {
      this.log("PMDSensorSettingHandle");
      const val = event?.target?.value;
      startECGRSLV(this.parseCtrlReply(val));
    };
    this.PMDCtrlChar?.addEventListener(
      "characteristicvaluechanged",
      PMDSensorSettingHandle,
      { once: true }
    );
    const cmd_buf = new Uint8Array(10);
    const cmd_buf_dataview = new DataView(cmd_buf.buffer);
    cmd_buf[0] = 2;
    cmd_buf[1] = 0;
    cmd_buf[2] = 1;
    cmd_buf[3] = 1;
    cmd_buf_dataview.setUint16(4, resolution, true);
    cmd_buf[6] = 0;
    cmd_buf[7] = 1;
    cmd_buf_dataview.setUint16(8, sample_rate, true);
    await this.writePMDCtrl(cmd_buf);
    const startReply = await this.awaitWithTimeout(
      startECGPromise,
      5e3,
      "Start ECG"
    );
    if (startReply?.error === ERROR_MSGS[0]) {
      this.ECGStarted = true;
    }
    return startReply;
  }
  async stopECG() {
    if (!this.ECGStarted) {
      return;
    }
    const endReply = await this.stopSensor(
      0
      /* ECG */
    );
    if (endReply?.error === ERROR_MSGS[0]) {
      this.ECGStarted = false;
    }
    return endReply;
  }
  async stopACC() {
    if (!this.ACCStarted) {
      return;
    }
    const endReply = await this.stopSensor(
      2
      /* ACC */
    );
    if (endReply?.error === ERROR_MSGS[0]) {
      this.ACCStarted = false;
    }
    return endReply;
  }
  async stopSensor(sensorType) {
    let endSensorRSLV;
    const endACCPromise = new Promise(
      (rslv) => {
        endSensorRSLV = rslv;
      }
    );
    const PMDSensorSettingHandle = (event) => {
      const val = event?.target?.value;
      endSensorRSLV(this.parseCtrlReply(val));
    };
    this.PMDCtrlChar?.addEventListener(
      "characteristicvaluechanged",
      PMDSensorSettingHandle,
      { once: true }
    );
    const cmd_buf = new Uint8Array(2);
    cmd_buf[0] = 3;
    cmd_buf[1] = sensorType;
    await this.writePMDCtrl(cmd_buf);
    return await this.awaitWithTimeout(endACCPromise, 5e3, "Stop Sensor");
  }
};

// sensor/driver.js
function createDriver(device, onPacketError = () => {
}, verbose = false) {
  let active = true;
  const listeners = [];
  const proxies = /* @__PURE__ */ new WeakMap();
  const wrap = (target) => {
    if (proxies.has(target)) return proxies.get(target);
    const proxy = new Proxy(target, {
      get(object, key) {
        if (key === "gatt") return wrap(object.gatt);
        if (key === "addEventListener") return (type, callback, options) => {
          if (!active) return;
          const guarded = (event) => {
            if (!active) return;
            try {
              callback(event);
            } catch (error) {
              onPacketError(error);
            }
          };
          listeners.push({ object, type, callback, guarded });
          object.addEventListener(type, guarded, options);
        };
        if (key === "removeEventListener") return (type, callback) => {
          for (const entry of listeners) {
            if (entry.object === object && entry.type === type && entry.callback === callback)
              object.removeEventListener(type, entry.guarded);
          }
        };
        if (["connect", "getPrimaryService", "getCharacteristic"].includes(key)) {
          return async (...args) => {
            if (!active) throw new Error("Connection cancelled");
            const result = await object[key](...args);
            if (!active) {
              device.gatt?.disconnect();
              throw new Error("Connection cancelled");
            }
            return wrap(result);
          };
        }
        const value = Reflect.get(object, key, object);
        if (typeof value !== "function") return value;
        return (...args) => {
          if (!active) throw new Error("Connection cancelled");
          return value.apply(object, args);
        };
      }
    });
    proxies.set(target, proxy);
    return proxy;
  };
  const driver = new PolarH10(wrap(device), verbose);
  return {
    driver,
    dispose() {
      active = false;
      for (const { object, type, guarded } of listeners) object.removeEventListener(type, guarded);
      listeners.length = 0;
      driver.heartRateHandleList.length = 0;
      for (const handlers of Object.values(driver.dataHandleDict)) handlers.length = 0;
    }
  };
}

// sensor/posture_acc_controller.js
var READING_BUCKET_MS = 100;
var ACC_RANGE_G = 4;
var ACC_RATE_HZ = 100;
var ACC_RESOLUTION_BITS = 16;
var PostureAccController = class {
  constructor({
    bluetooth = globalThis.navigator?.bluetooth,
    makeDriver = createDriver,
    onChange = () => {
    },
    onReading = () => {
    },
    retryBaseMs = 1e3,
    maxRetries = 5,
    timeoutMs = 2e4
  } = {}) {
    Object.assign(this, { bluetooth, makeDriver, onChange, onReading, retryBaseMs, maxRetries, timeoutMs });
    this.state = "idle";
    this.message = "No Polar H10 connected.";
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
      this.scheduleRetry("Sensor disconnected.");
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
    if (!this.bluetooth) return this.status("error", "Web Bluetooth requires Chrome or Edge on a supported platform, using localhost or HTTPS.");
    this.busy = true;
    const generation = ++this.generation;
    this.status("connecting", "Select a Polar H10 in the Bluetooth chooser.");
    try {
      const device = await this.bluetooth.requestDevice({ filters: [{ namePrefix: "Polar H10" }], optionalServices: SERVICES });
      console.log(`[posture-ble] requestDevice resolved: name=${device.name} id=${device.id}`);
      if (generation !== this.generation) return;
      this.device = device;
      this.device.addEventListener("gattserverdisconnected", this.lost);
      this.wanted = true;
      this.attempt = 0;
    } catch (error) {
      if (generation === this.generation) this.status("error", error.name === "NotFoundError" ? "No device selected. Connect to try again." : `Bluetooth selection failed: ${error.message}`);
    } finally {
      this.busy = false;
    }
    if (generation === this.generation && this.wanted) await this.open(generation);
  }
  async open(generation) {
    if (this.busy || !this.wanted || generation !== this.generation) return;
    this.busy = true;
    this.status(this.attempt ? "reconnecting" : "connecting", "Initializing Polar H10\u2026");
    let failure;
    let timer;
    const scope = this.makeDriver(this.device, () => {
      this.packetErrors = (this.packetErrors || 0) + 1;
    }, true);
    this.scope = scope;
    const current = () => this.wanted && generation === this.generation && this.scope === scope;
    const check = () => {
      if (!current() || !this.device.gatt?.connected) throw new Error("Sensor disconnected or connection cancelled");
    };
    try {
      const setup = async () => {
        const p = scope.driver;
        p.addEventListener("ACC", (frame) => {
          if (current()) this._handleAccFrame(frame);
        });
        console.log("[posture-ble] init() starting");
        const t0 = performance.now();
        await p.init(100);
        console.log(`[posture-ble] init() done in ${(performance.now() - t0).toFixed(0)}ms`);
        check();
        await new Promise((resolve) => setTimeout(resolve, 400));
        check();
        console.log("[posture-ble] startACC() starting");
        const t1 = performance.now();
        const reply = await p.startACC(ACC_RANGE_G, ACC_RATE_HZ, ACC_RESOLUTION_BITS);
        console.log(`[posture-ble] startACC() done in ${(performance.now() - t1).toFixed(0)}ms, reply=`, reply);
        if (!["SUCCESS", "ALREADY IN STATE"].includes(reply?.error))
          throw new Error(reply?.error || "No ACC start acknowledgement");
        check();
      };
      await Promise.race([setup(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Connection timed out")), this.timeoutMs);
      })]);
      if (current()) {
        this._sessionStartMs = null;
        this._bucketIndex = null;
        this.status("connected", `Connected to ${this.device.name || "Polar H10"}. Streaming accelerometer data.`);
      }
    } catch (error) {
      failure = error;
      if (current()) {
        this.release();
        this.device.gatt?.disconnect();
      }
    } finally {
      clearTimeout(timer);
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
      const ax = samples[i * 3] / 1e3;
      const ay = samples[i * 3 + 1] / 1e3;
      const az = samples[i * 3 + 2] / 1e3;
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
          t: this._bucketIndex * READING_BUCKET_MS / 1e3,
          ax: this._bucketSumAx / this._bucketCount,
          ay: this._bucketSumAy / this._bucketCount,
          az: this._bucketSumAz / this._bucketCount
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
      this.status("error", `${reason} Reconnection attempts exhausted. Check battery/range and OS pairing (firmware 4+), then Connect again.`);
      return;
    }
    const delay = Math.min(this.retryBaseMs * 2 ** this.attempt, 16e3);
    this.attempt++;
    this.status("reconnecting", `${reason} Retrying ${this.attempt}/${this.maxRetries} in ${(delay / 1e3).toFixed(1)}s.`);
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
    this.busy = false;
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.release();
    this.device?.removeEventListener("gattserverdisconnected", this.lost);
    this.device?.gatt?.disconnect();
    this.status("idle", "No Polar H10 connected.");
  }
};

// sensor/posture_entry.js
(function installConsoleMirror() {
  const buffer = [];
  let flushTimer = null;
  function flush() {
    flushTimer = null;
    if (!buffer.length) return;
    const lines = buffer.splice(0, buffer.length);
    fetch("/api/posture/debug-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lines })
    }).catch(() => {
    });
  }
  function mirror(kind) {
    const original = console[kind].bind(console);
    console[kind] = (...args) => {
      original(...args);
      buffer.push(`[${kind}] ${args.map((a) => typeof a === "string" ? a : JSON.stringify(a)).join(" ")}`);
      if (!flushTimer) flushTimer = setTimeout(flush, 150);
    };
  }
  mirror("log");
  mirror("warn");
  mirror("error");
})();
var STATE_LABELS = {
  idle: "Not connected",
  connecting: "Connecting\u2026",
  reconnecting: "Reconnecting\u2026",
  error: "Connection error"
};
async function post(url, body) {
  try {
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  } catch {
  }
}
function mountPosturePolar(root2) {
  const connectBtn = root2.querySelector("#postureConnectBtn");
  const disconnectBtn = root2.querySelector("#postureDisconnectBtn");
  const statusEl = root2.querySelector("#postureBleStatus");
  const startBtn = document.getElementById("postureStartBtn");
  if (!connectBtn || !disconnectBtn || !statusEl) return;
  const controller = new PostureAccController({
    onChange(c) {
      const connected = c.state === "connected";
      statusEl.textContent = connected ? `Connected (${c.device?.name || "Polar H10"})` : STATE_LABELS[c.state] || c.message;
      connectBtn.disabled = c.busy || c.wanted;
      disconnectBtn.disabled = !c.wanted && !c.busy;
      if (startBtn) startBtn.disabled = !connected;
      post("/api/posture/connection", { connected });
    },
    onReading(reading) {
      post("/api/posture/reading", reading);
    }
  });
  connectBtn.addEventListener("click", () => void controller.connect());
  disconnectBtn.addEventListener("click", () => controller.disconnect());
}
var root = document.getElementById("posturePolarPanel");
if (root) mountPosturePolar(root);
