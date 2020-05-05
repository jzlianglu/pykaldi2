import os
import sys
import editdistance

#def main():
if __name__ == '__main__':
  #  parser = argparse.ArgumentParser()                                                                                 
  #  parser.add_argument("-config")                                                                                     
    hyp_path = sys.argv[1]
    ref_path = sys.argv[2]

    if not os.path.isfile(hyp_path):
       print("File path {} does not exist. Exiting...".format(hyp_path))
       sys.exit()

    if not os.path.isfile(ref_path):
       print("File path {} does not exist. Exiting...".format(ref_path))
       sys.exit()

    dict_ref = dict()
    with open(ref_path) as f:
        for line in f:
            line = line.split()
            key = line.pop(0)
            dict_ref[key] = line
        
    f.close()   
  
    with open(hyp_path) as f:
        for line in f:
            line = line.split()
            key = line.pop(0)

            if(len(line) > 0):
                err = editdistance.eval(line, dict_ref[key])
            else:
                err = len(dict_ref[key])

            print("reco:", key, ' '.join(line))
            print("ref:", key, ' '.join(dict_ref[key]))
            print(key, "err:", err, "ref_len:", len(dict_ref[key]))


    f.close() 
