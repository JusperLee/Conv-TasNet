def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict= dict()
    line = 0
    lines = open(scp_path,'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line+=1
        if len(scp_parts) != 2 :
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key,value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                    key, scp_path))
        
        scp_dict[key] = value
    
    return scp_dict


if __name__ == "__main__":
    print(len(handle_scp('/home/likai/data1/create_scp/cv_s2.scp')))

        
