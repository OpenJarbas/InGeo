from json_database import JsonStorageXDG


class LocationFingerprints(JsonStorageXDG):
    def __init__(self, family):
        super().__init__(family, subfolder="InGeo")
        if "locations" not in self:
            self["locations"] = {}
        if "sensors" not in self:
            self["sensors"] = {}
        if "fingerprints" not in self:
            self["fingerprints"] = {}

    def clear(self):
        super(LocationFingerprints, self).clear()
        if "locations" not in self:
            self["locations"] = {}
        if "sensors" not in self:
            self["sensors"] = {}
        if "fingerprints" not in self:
            self["fingerprints"] = {}

    def reload(self):
        super(LocationFingerprints, self).reload()
        if "locations" not in self:
            self["locations"] = {}
        if "sensors" not in self:
            self["sensors"] = {}
        if "fingerprints" not in self:
            self["fingerprints"] = {}

    def sensor2fingerprint(self, location_name, sensor_data):
        row = [0] * len(self["sensors"])
        location_id = self["locations"][location_name]
        for k, v in sensor_data.items():
            sensor_id = int(self["sensors"][k])
            row[sensor_id] = float(v)
        return [int(location_id)] + row

    def add_fingerprint(self, location_name, sensor_data):
        if location_name not in self["locations"]:
            self["locations"][location_name] = len(self["locations"])

        for k in sensor_data.keys():
            if k not in self["sensors"]:
                self["sensors"][k] = len(self["sensors"])

        if location_name not in self["fingerprints"]:
            self["fingerprints"][location_name] = []
        fingerprint = self.sensor2fingerprint(location_name, sensor_data)
        self["fingerprints"][location_name].append(fingerprint)

    def import_csv(self, csv_path):
        with open(csv_path) as f:
            lines = f.read().split("\n")
        header = lines[0].split(",")
        for l in lines[1:]:
            try:
                vals = l.split(",")
                name = vals[0]
                sensor_data = {header[i]: vals[i] or '0'
                               for i in range(1, len(header))}
                self.add_fingerprint(name, sensor_data)
            except:
                pass

    def export_csv(self, csv_path):
        with open(csv_path, "w") as f:
            keys = [""] * len(self["sensors"])
            for s, sid in self["sensors"].items():
                keys[sid] = s
            keys.insert(0, "name")
            header = ",".join(keys)
            f.write(header + "\n")
            for location, data in self["fingerprints"].items():
                for datapoint in data:
                    row = [0] * len(datapoint)
                    for sensor_id, val in enumerate(datapoint):
                        keys[sensor_id] = self["sensors"]
                        row[sensor_id] = str(val)
                    row[0] = location
                    row = ",".join(row)
                    f.write(row + "\n")


if __name__ == "__main__":
    fpdb = LocationFingerprints("test")
    csv = "/home/user/PycharmProjects/find3lib/test/reverse.csv"
    fpdb.import_csv(csv)
    fpdb.export_csv('../test/test.csv')
    fpdb.store()
    #print(fpdb["fingerprints"])
