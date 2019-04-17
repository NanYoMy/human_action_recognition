import glob
import shutil
skeleton_set=glob.glob('D:\\SBU\\*\\*\\*\\*\\*.txt')
for s in skeleton_set:
    token=s.split("\\")
    filename=token[4]+"-"+token[3]+"-"+token[5]+token[6]

    shutil.copy(s,'.\\data\\skeleton5\\'+filename)
