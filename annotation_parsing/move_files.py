import os
import shutil
sourcepath='B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\False\\true'
sourcefiles = os.listdir(sourcepath)
destinationpath = 'B:\\Desktop\\FER\\DIPL_1\\RaspoznavanjeUzoraka\\haralick-features-detector\\annotation_parsing\\False\\false'
for file in sourcefiles:
    if file.endswith('false.png'):
        shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))