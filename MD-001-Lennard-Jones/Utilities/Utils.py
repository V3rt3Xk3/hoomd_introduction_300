import os
import gsd.hoomd

class GSDUtils:

    @staticmethod
    def saveSnapshot(fileName, gsdSnapshot, hoomdSnapshot):
        # WOW: This is superwierd, I couldn't find a good way to copy a hoomd.Snapshot to gsd.hoomd.Snapshot. The only difference seems like, there is a validate method for the properties and the  class itself.
        gsdSnapshot.particles.N = hoomdSnapshot.particles.N
        gsdSnapshot.particles.position = hoomdSnapshot.particles.position
        gsdSnapshot.particles.orientation = hoomdSnapshot.particles.orientation
        gsdSnapshot.particles.typeid = hoomdSnapshot.particles.typeid
        gsdSnapshot.particles.types = hoomdSnapshot.particles.types
        gsdSnapshot.configuration.box = hoomdSnapshot.configuration.box

        # NOTE: Saving the snapshot
        currentPath = os.path.dirname(__file__)
        snapshotPath = SysUtils.generateSnaptshotPathFromUtils(currentPath=currentPath, fileName=fileName)
        with gsd.hoomd.open(name=snapshotPath, mode='xb') as file:
            file.append(gsdSnapshot)


class SysUtils:

    @staticmethod
    def generateSnaptshotPath(currentPath, fileName):
        # HACK: Generating the savePath

        if (currentPath == ""):
            file_path = "./snapshots/" + fileName
        elif (currentPath != ""):
            file_path = currentPath + '/snapshots/' + fileName

        return file_path
    
    @staticmethod
    def generateSnaptshotPathFromUtils(currentPath, fileName):
        # HACK: Generating the savePath

        if (currentPath == ""):
            file_path = "./../snapshots/" + fileName
        elif (currentPath != ""):
            file_path = currentPath + '/../snapshots/' + fileName

        return file_path