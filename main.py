import autokeras as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
from urllib.request import urlopen
import zipfile
import os
import json


def download_map(maphash):
    response = urlopen("https://beatsaver.com/api/maps/hash/" + maphash)
    data_json = json.loads(response.read())

    zip_url = data_json["versions"][0]["downloadURL"]
    filehandle, _ = urllib.request.urlretrieve(zip_url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    for filename in zip_file_object.namelist():
        file = zip_file_object.open(filename)
        content = file.read()

        newfilename = "maps/" + maphash + "/" + filename
        os.makedirs(os.path.dirname(newfilename), exist_ok=True)

        file2 = open(newfilename, "wb")
        file2.write(content)
        file2.close()


def loadPlist(pathToPlist):
    f = open(pathToPlist)
    plist = json.load(f)
    f.close()

    count = 0
    for song in plist['songs']:
        count += len(song["hitbloq"]["difficulties"]["Standard"])

    xData = np.ones([count, 5, 5, 3], dtype=float)
    yData = np.ones([count, 1], dtype=float)

    index = 0

    for song in plist['songs']:
        info = {}
        folderName = "maps/" + song["hash"]
        if not os.path.exists(folderName):
            download_map(song["hash"])
        if os.path.exists(folderName + "/Info.dat"):
            f = open(folderName + "/Info.dat")
            info = json.load(f)
            f.close()
        elif os.path.exists(folderName + "/info.dat"):
            f = open(folderName + "/info.dat")
            info = json.load(f)
            f.close()
        else:
            continue

        if os.path.exists(folderName + "/" + info["_songFilename"]):
            os.remove(folderName + "/" + info["_songFilename"])
        if os.path.exists(folderName + "/" + info["_coverImageFilename"]):
            os.remove(folderName + "/" + info["_coverImageFilename"])

        for diff in info["_difficultyBeatmapSets"]:
            characteristic = diff["_beatmapCharacteristicName"]
            if characteristic not in song["hitbloq"]["difficulties"]:
                continue

            songdiffs = song["hitbloq"]["difficulties"][characteristic]

            for diffMap in diff["_difficultyBeatmaps"]:
                ihatethisformat = diffMap["_difficulty"][0].lower() + diffMap["_difficulty"][1:]
                if ihatethisformat not in songdiffs:
                    continue

                f = open(folderName + "/" + diffMap["_beatmapFilename"], encoding="utf8")
                mape = json.load(f)
                f.close()
                yData[index] = songdiffs[ihatethisformat]

                x = 0
                y = 0
                for note in mape["_notes"]:
                    xData[index][x][y][0] = note["_time"]
                    xData[index][x][y][1] = note["_lineIndex"] * 1000 + note["_lineLayer"] * 100 + note["_type"] * 10 + \
                                            note["_cutDirection"]
                    xData[index][x][y][2] = diffMap["_noteJumpMovementSpeed"]

                    x += 1
                    if x == 100:
                        x = 0
                        y += 1

                index += 1
    return xData, yData


def loadPlistProd(pathToPlist):
    f = open(pathToPlist)
    plist = json.load(f)
    f.close()

    result = []
    names = []

    for song in plist['songs']:
        info = {}
        folderName = "maps/" + song["hash"]
        if not os.path.exists(folderName):
            download_map(song["hash"])
        if os.path.exists(folderName + "/Info.dat"):
            f = open(folderName + "/Info.dat")
            info = json.load(f)
            f.close()
        elif os.path.exists(folderName + "/info.dat"):
            f = open(folderName + "/info.dat")
            info = json.load(f)
            f.close()
        else:
            continue

        if os.path.exists(folderName + "/" + info["_songFilename"]):
            os.remove(folderName + "/" + info["_songFilename"])
        if os.path.exists(folderName + "/" + info["_coverImageFilename"]):
            os.remove(folderName + "/" + info["_coverImageFilename"])

        for diff in info["_difficultyBeatmapSets"]:
            characteristic = diff["_beatmapCharacteristicName"]

            for diffMap in diff["_difficultyBeatmaps"]:
                ihatethisformat = diffMap["_difficulty"][0].lower() + diffMap["_difficulty"][1:]

                f = open(folderName + "/" + diffMap["_beatmapFilename"], encoding="utf8")
                mape = json.load(f)
                f.close()

                groups = {}
                groupList = []

                for note in mape["_notes"]:
                    if note["_type"] == 0 or note["_type"] == 1:
                        if note["_time"] in groups:
                            groups[note["_time"]].append(note)
                        else:
                            groups[note["_time"]] = [note]

                for key in groups:
                    groupList.append((key, groups[key]))

                processed = []

                """
                int start = 0;
                int finish = groupList.Count() > 13 ? 13 : groupList.Count();

                do {
                    var window = new List<List<float>>();
                    for (int i = start; i < finish; i++) {

                        (float, List <NoteStruct>)? lastGroup = i == 0 ? null : groupList[i - 1];
                        List<float> list = new List<float>();
                        for (int j = 0; j < 13; j++) { list.Add(0); }

                        if (groupList[i].Item2.Count > 3) {
                            var p = 3;
                        }

                        foreach (var item in groupList[i].Item2.OrderBy(n => n.noteEvent.noteID))
                        {
                            NoteParams param = new NoteParams(item.noteEvent.noteID);

                            int id = (param.scoringType == ScoringType.Default ? 3 : (int)param.scoringType) * 100 + param.cutDirection * 10 + param.colorType;

                            list[param.noteLineLayer * 4 + param.lineIndex] = ((float)id) / 1000.0f;
                        }

                        list[12] = lastGroup != null ? (groupList[i].Item1 - lastGroup.Value.Item1) : groupList[i].Item1;

                        lastGroup = groupList[i];

                        window.Add(list);
                    }

                    resultX.Add(window);

                    resultY.Add((groupList[finish - 1].Item2.Average(n => n.accuracy) - groupList[start].Item2.Average(n => n.accuracy) + 1.0f) / 5.0f);

                    if (finish >= groupList.Count()) {
                        break;
                    } else {
                        start += 4;
                        finish += 4;
                        if (finish > groupList.Count()) {
                            finish = groupList.Count();
                        }
                    }

                } while (true);
                """

                start = 0
                finish = 13 if len(groupList) > 13 else len(groupList)

                while True:
                    window = []
                    for i in range(start, finish):
                        lastGroup = None if i == 0 else groupList[i - 1]

                        list = []
                        for j in range(0, 13):
                            list.append(0)

                        for item in groupList[i][1]:
                            list[item["_lineLayer"] * 4 + item["_lineIndex"]] = (300 + item["_cutDirection"] * 10 +
                                                                                 item["_type"]) / 1000

                        list[12] = groupList[i][0] - lastGroup[0] if lastGroup is not None else groupList[i][0]

                        window.append(list)

                    processed.append(window)

                    if finish >= len(groupList):
                        break
                    else:
                        start += 4
                        finish += 4
                        if finish > len(groupList):
                            finish = len(groupList)

                xData = np.ones([len(processed), 13, 13, 1], dtype=float)

                index = 0
                for p in processed:
                    for i in range(0, 13):
                        for j in range(0, 13):
                            if j < len(p[i]):
                                xData[index][i][j][0] = p[i][j]
                            else:
                                xData[index][i][j][0] = 0
                    index += 1

                result.append(xData)
                names.append(info["_songName"] + "-" + ihatethisformat + "   " + characteristic)

    return result, names


# f = open('scoresaberduh.bplist')
# plist = list(map(lambda n: n["hash"], json.load(f)['songs']))
# f.close()
#
# fortest = []
#
# for i in range(1, 15):
#     f = open('ranked_star_0' + str(i) + ".json")
#     ranked = json.load(f)
#     for map in ranked["songs"]:
#         if map["hash"] not in plist:
#             fortest.append(map["hash"])
#     f.close()
#
# newplist = {"songs": []}
# for hash in fortest:
#     try:
#         response = urlopen("https://beatsaver.com/api/maps/hash/" + hash)
#         data_json = json.loads(response.read())
#
#         difficulties = {}
#         toadd = False
#         for diff in data_json["versions"][0]["diffs"]:
#             if "stars" in diff and diff["stars"] > 0:
#                 toadd = True
#                 ihatethisformat = diff["difficulty"][0].lower() + diff["difficulty"][1:]
#                 if diff["characteristic"] not in difficulties:
#                     difficulties[diff["characteristic"]] = {}
#                 difficulties[diff["characteristic"]][ihatethisformat] = diff["stars"]
#
#         if toadd:
#             mape = {"hitbloq": {"difficulties": difficulties}, "hash": hash}
#         newplist["songs"].append(mape)
#     except:
#         print("")
#
#
#
#
# with open('test.plist', 'w') as outfile:
#     json_string = json.dumps(newplist)
#     json.dump(json_string, outfile)

x_train = np.load('C:\\Users\\vikto\\Documents\\replays\\xs.npy')
y_train = np.load('C:\\Users\\vikto\\Documents\\replays\\ys.npy')

x_test = np.load('C:\\Users\\vikto\\Documents\\replays2\\xs.npy')
y_test = np.load('C:\\Users\\vikto\\Documents\\replays2\\ys.npy')

model = ak.ImageRegressor(overwrite=True, max_trials=5)

model.fit(
    x_train,
    y_train,
    epochs=5)
exported = model.export_model()

try:
    exported.save("model_autokeras", save_format="tf")
except Exception:
    exported.save("model_autokeras.h5")

# Evaluate the best model with testing data.
print(model.evaluate(x_test, y_test))

notRanked = loadPlistProd('test.bplist')
loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

for i in range(0, len(notRanked[0])):
    predicted_y = loaded_model.predict(notRanked[0][i])

    acc = 1.0
    for y in predicted_y:
        acc += (y - 1.0)

    print("Stars?: " + str(acc * 4) + " Mape: " + notRanked[1][i])

# for i in range(188, 210):
#     url = "https://scoresaber.com/api/leaderboards?category=1&maxStar=50&minStar=0&page=" + str(i) + "&ranked=1&sort=0&verified=0"
#     req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
#     response = urllib.request.urlopen(req)
#     data_json = json.loads(response.read())
#
#     for l in data_json["leaderboards"]:
#         hash = l["songHash"]
#         diff = l["difficulty"]["difficulty"]
#         date = l["rankedDate"]
#
#         if date is not None:
#             url = "https://localhost:7040/map/rdate/" + hash + "?date=" + date + "&diff=" + str(diff)
#             urlopen(url)




