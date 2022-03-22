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

    xData = np.ones([count, 100, 100, 3], dtype=float)
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
                    xData[index][x][y][1] = note["_lineIndex"] * 1000 + note["_lineLayer"] * 100 + note["_type"] * 10 + note["_cutDirection"]
                    xData[index][x][y][2] = diffMap["_noteJumpMovementSpeed"]

                    x += 1
                    if x == 100:
                        x = 0
                        y += 1

                index += 1
    return xData, yData


(x_train, y_train) = loadPlist('scoresaberduh.bplist')
(x_test, y_test) = loadPlist('test.bplist')

model = ak.ImageRegressor(overwrite=True, max_trials=40)

model.fit(x_train, y_train, epochs=20)
exported = model.export_model()

predicted_y = model.predict(x_test)

f = open('test.bplist')
plist = json.load(f)
f.close()

index = 0
for map in plist["songs"]:
    for diff in map["hitbloq"]["difficulties"]["Standard"]:
        print(map["hash"] + " " + diff + " c:" + str(map["hitbloq"]["difficulties"]["Standard"][diff]) + "p:" + str(predicted_y[index]))
        index += 1

try:
    exported.save("model_autokeras", save_format="tf")
except Exception:
    exported.save("model_autokeras.h5")

# Evaluate the best model with testing data.
print(model.evaluate(x_test, y_test))




