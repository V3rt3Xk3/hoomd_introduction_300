


class SysUtils:

    @staticmethod
    def generateSnaptshotPath(currentPath, fileName):
        # HACK: Generating the savePath

        if (currentPath == ""):
            file_path = "./snapshots/" + fileName
        elif (currentPath != ""):
            file_path = currentPath + '/snapshots/' + fileName

        return file_path