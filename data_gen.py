import csv
import numpy as np
import setting as st

# seed = st.random_seed
# np.random.seed(seed)


def network_info(datapath):
    # f = open('data/node_info.csv', 'r', encoding='UTF8')
    f = open('data/node_info_final.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    newnodeid = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    nid = 0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            newnodeid[int(line[0])] = nid
            node_data[nid] = {'NODE_ID': nid, 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(line[5]), 'NODE_ID_OLD': float(line[0])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            nid += 1
            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])
        linenum += 1

    print('total nodes', linenum - 1)
    f.close()


    # f = open('data/link_info.csv', 'r', encoding='UTF8')
    f = open('data/link_info_final.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    lid = 0
    linenum = 0
    link_data = {}
    newlinkid = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            fn = newnodeid[int(line[1])]
            tn = newnodeid[int(line[2])]

            newlinkid[int(line[0])] = lid
            link_data[lid] = {'LINK_ID': lid, 'F_NODE': fn, 'T_NODE': tn,
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
            lid += 1
        linenum += 1
    print('total links', linenum-1)
    f.close()



    f = open(datapath, 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_traffic = {}
    for line in rdr:
        linenum += 1

        if int(line[1]) > 4000000000 and int(line[1]) < 4090000000:

            if int(line[1]) not in newlinkid.keys():
                continue
            lid = newlinkid[int(line[1])]
            # print(lid, int(line[1]))
            if lid in link_traffic:
                link_traffic[lid].append(float(line[2]))
            else:
                link_traffic[lid] = [float(line[2])]
            # print(line)
    # print('l', linenum)
    f.close()





    f = open('data/chargingstation.csv', 'r')
    rdr = csv.reader(f)
    linenum = 0
    cs_info = {}
    for line in rdr:
        if linenum == 0:
            print(line)
        else:
            if line[7] == 'Y':
                mindist = 100000
                n_id = -1
                # print(linenum, line[7],line[17],line[16], line)
                x1 = float(line[17])
                y1 = float(line[16])

                for n in node_data.keys():
                    x2 = node_data[n]['long']
                    y2 = node_data[n]['lat']
                    diff = abs(x1 - x2) + abs(y1 - y2)
                    if mindist > diff:
                        mindist = diff
                        n_id = n
                # print(n_id, mindist )
                if n_id in cs_info.keys():
                    if diff < cs_info[n_id]['diff_node']:
                        cs_info[n_id] = {'CS_ID': n_id, 'CS_NAME': line[0], 'lat': node_data[n_id]['lat'], 'long': node_data[n_id]['long'],'real_lat': float(line[16]),
                                         'real_long': float(line[17]), 'diff_node': mindist}
                else:
                    cs_info[n_id] = {'CS_ID': n_id, 'CS_NAME': line[0], 'lat': node_data[n_id]['lat'],
                                    'long': node_data[n_id]['long'], 'real_lat': float(line[16]),
                                    'real_long': float(line[17]), 'diff_node': mindist}


        linenum+=1
    f.close()
    # print(len(cs_info))
    # for n_id in cs_info.keys():
    #     print(cs_info[n_id])




    # f = open('data/node_info_all_final.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in node_data.keys():
    #     wr.writerow([node_data[n_id]['NODE_ID'],node_data[n_id]['NODE_TYPE'],node_data[n_id]['NODE_NAME'],node_data[n_id]['lat'],node_data[n_id]['long'],node_data[n_id]['NODE_ID_OLD']])
    # f.close()
    #
    #
    # f = open('data/link_info_all_final.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in link_data.keys():
    #     wr.writerow([link_data[n_id]['LINK_ID'],link_data[n_id]['F_NODE'],link_data[n_id]['T_NODE'],link_data[n_id]['MAX_SPD'],link_data[n_id]['LENGTH'],link_data[n_id]['CUR_SPD'], link_data[n_id]['WEIGHT']])
    # f.close()
    #
    # f = open('data/cs_info_all_final.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in cs_info.keys():
    #     wr.writerow([cs_info[n_id]['CS_ID'],cs_info[n_id]['CS_NAME'],cs_info[n_id]['lat'],cs_info[n_id]['long'],cs_info[n_id]['real_lat'],cs_info[n_id]['real_long']])
    # f.close()


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy

def network_info_jejusi():
    f = open('data/node_info_jejusi.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    newnodeid = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    nid = 0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[3]),
                                       'long': float(line[4])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'

        linenum += 1

    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_info_jejusi.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)

    linenum = 0
    link_data = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            fnode = int(line[1])
            tnode = int(line[2])
            # print(fnode, tnode, fnode in node_data.keys(),  tnode in node_data.keys())

            if fnode in node_data.keys() and tnode in node_data.keys():
                link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                                       'MAX_SPD': float(line[3]), 'LENGTH': float(line[4])/1000, 'CUR_SPD': float(
                                    line[5]), 'WEIGHT': float(line[6])}


            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])

        linenum += 1

    print('total links', len(link_data))
    f.close()

    # f = open('data/link_info_test.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in link_data.keys():
    #     wr.writerow([link_data[n_id]['LINK_ID'],link_data[n_id]['F_NODE'],link_data[n_id]['T_NODE'],link_data[n_id]['MAX_SPD'],link_data[n_id]['LENGTH'],link_data[n_id]['CUR_SPD'], link_data[n_id]['WEIGHT']])
    # f.close()


    # f = open('data/cs_info_jejusi.csv', 'r')
    f = open('data/cs_info_jejusi_55.csv', 'r')
    rdr = csv.reader(f)

    linenum = 0
    cs_info = {}
    for line in rdr:
        if linenum == 0:
            print(line)
        else:

            cs_info[int(line[0])] = {'CS_ID': int(line[0]), 'CS_NAME': line[1], 'lat': float(line[2]), 'long': float(line[3]),
                            'real_lat': float(line[4]), 'real_long': float(line[5])}

        linenum+=1
    f.close()

    print('num of cs: ', len(cs_info))

    link_traffic = {}
    for l in link_data.keys():
        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(int(maxspd - maxspd * 0.4), maxspd, 288))


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy

def network_info_jejusi_77():
    f = open('data/node_jeju_topo_77.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    newnodeid = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    nid = 0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': 0, 'NODE_NAME': 0,
                                       'lat': float(line[6])/100000,
                                       'long': float(line[5])/100000}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'


            if minx > float(line[5])/100000:
                minx = float(line[5])/100000
            if miny > float(line[6])/100000:
                miny = float(line[6])/100000
            if maxx < float(line[5])/100000:
                maxx = float(line[5])/100000
            if maxy < float(line[6])/100000:
                maxy = float(line[6])/100000


        linenum += 1

    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_jeju_topo_254.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)

    linenum = 0
    link_data = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            fnode = int(line[1])
            tnode = int(line[2])
            # print(fnode, tnode, fnode in node_data.keys(),  tnode in node_data.keys())

            if fnode in node_data.keys() and tnode in node_data.keys():
                link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                                       'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])*1.6/1000,
                                           'CUR_SPD': float(line[11]), 'WEIGHT': float(line[11])}


            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])

        linenum += 1

    print('total links', len(link_data))
    f.close()

    # f = open('data/link_info_test.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in link_data.keys():
    #     wr.writerow([link_data[n_id]['LINK_ID'],link_data[n_id]['F_NODE'],link_data[n_id]['T_NODE'],link_data[n_id]['MAX_SPD'],link_data[n_id]['LENGTH'],link_data[n_id]['CUR_SPD'], link_data[n_id]['WEIGHT']])
    # f.close()


    # f = open('data/cs_info_jejusi.csv', 'r')

    cs_info = {}
    # cs_location_list = [5, 19, 23, 39, 52]
    cs_location_list = [11, 25, 32]
    # cs_location_list = [11, 17, 30, 31, 64]
    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                             'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                             'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                             'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                             'real_long': node_data[csid]['long'], 'diff_node': 0}


    print('num of cs: ', len(cs_info))

    link_traffic = {}
    for l in link_data.keys():
        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(int(maxspd - maxspd * 0.1), maxspd, 288))


    # return link_data, node_data, link_traffic, cs_info, 140.7, 39.55, 141.02, 39.675
    return link_data, node_data, link_traffic, cs_info,  minx, miny, maxx, maxy


def network_info_34():

    # np.random.seed(seed)

    f = open('data/node_34.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    newnodeid = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    nid = 0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': 0, 'NODE_NAME': 0,
                                       'lat': float(line[6]),
                                       'long': float(line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'


            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])


        linenum += 1

    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_34.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)

    linenum = 0
    link_data = {}
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            fnode = int(line[1])
            tnode = int(line[2])
            # print(fnode, tnode, fnode in node_data.keys(),  tnode in node_data.keys())

            if fnode in node_data.keys() and tnode in node_data.keys():
                link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                                       'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])*1.5/1000,
                                           'CUR_SPD': float(line[11]), 'WEIGHT': float(line[11])}


            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])

        linenum += 1

    print('total links', len(link_data))
    f.close()

    # f = open('data/link_info_test.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in link_data.keys():
    #     wr.writerow([link_data[n_id]['LINK_ID'],link_data[n_id]['F_NODE'],link_data[n_id]['T_NODE'],link_data[n_id]['MAX_SPD'],link_data[n_id]['LENGTH'],link_data[n_id]['CUR_SPD'], link_data[n_id]['WEIGHT']])
    # f.close()


    # f = open('data/cs_info_jejusi.csv', 'r')

    cs_info = {}
    # cs_location_list = [8, 11, 24, 27]
    cs_location_list = [8, 11, 25]

    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                             'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                             'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                             'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                             'real_long': node_data[csid]['long'], 'diff_node': 0}


    print('num of cs: ', len(cs_info))

    link_traffic = {}
    for l in link_data.keys():
        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(int(maxspd - maxspd * 0.1), maxspd, 288))


    # return link_data, node_data, link_traffic, cs_info, 140.7, 39.55, 141.02, 39.675
    return link_data, node_data, link_traffic, cs_info,  minx, miny, maxx, maxy


def network_info_jejusi_small():
    f = open('data/node_info_jejusi_small.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    newnodeid = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    nid = 0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            newnodeid[int(line[0])] = nid
            node_data[nid] = {'NODE_ID': nid, 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[3]),
                                       'long': float(line[4]),'NODE_ID_OLD': float(line[0])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            nid += 1
        linenum += 1

    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_info_jejusi.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)

    linenum = 0
    lid = 0
    link_data = {}
    newlinkid = {}





    for line in rdr:
        if linenum == 0:
            a = line
        else:

            if int(line[1]) in newnodeid.keys() and int(line[2]) in newnodeid.keys() :
                fnode = newnodeid[int(line[1])]
                tnode = newnodeid[int(line[2])]
                print(fnode, newnodeid[int(line[1])])
                print(tnode, newnodeid[int(line[2])])


                newlinkid[int(line[0])] = lid




            # if fnode in node_data.keys() and tnode in node_data.keys():
                link_data[lid] = {'LINK_ID': lid, 'F_NODE': fnode, 'T_NODE': tnode,
                                                       'MAX_SPD': float(line[3]), 'LENGTH': float(line[4])/1000, 'CUR_SPD': float(
                                    line[5]), 'WEIGHT': float(line[6])}
                lid += 1


            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])

        linenum += 1











    print('total links', len(link_data))
    f.close()

    # f = open('data/link_info_test.csv', 'w', newline='')
    # wr = csv.writer(f)
    # for n_id in link_data.keys():
    #     wr.writerow([link_data[n_id]['LINK_ID'],link_data[n_id]['F_NODE'],link_data[n_id]['T_NODE'],link_data[n_id]['MAX_SPD'],link_data[n_id]['LENGTH'],link_data[n_id]['CUR_SPD'], link_data[n_id]['WEIGHT']])
    # f.close()


    # f = open('data/cs_info_jejusi.csv', 'r')
    # f = open('data/cs_info_jejusi_10.csv', 'r')
    f = open('data/cs_info_jejusi_5.csv', 'r')
    rdr = csv.reader(f)

    linenum = 0
    cs_info = {}
    for line in rdr:
        if linenum == 0:
            print(line)
        else:
            oldcsid = int(line[0])
            nid=0
            while node_data[nid]['NODE_ID_OLD'] != oldcsid:
                nid+=1

            cs_info[nid] = {'CS_ID': nid, 'CS_NAME': line[1], 'lat': float(line[2]), 'long': float(line[3]),
                            'real_lat': float(line[4]), 'real_long': float(line[5]), 'CS_ID_OLD':int(line[0])}

        linenum+=1
    f.close()

    print('num of cs: ', len(cs_info))

    link_traffic = {}
    for l in link_data.keys():
        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(int(maxspd - maxspd * 0.4), maxspd, 288))


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy


def network_info_simple_39():

    f = open('data/node_info_39.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(
                                           line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            # print((line[5]), (line[6]))
            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])
        linenum += 1
    print('total nodes', linenum - 1)
    f.close()


    # f = open('data/link_info_39.csv', 'r', encoding='UTF8')
    f = open('data/link_info_39_reverse.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_data = {}

    for line in rdr:
        if linenum == 0:
            a = line
        else:

            link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()

    cs_info = {}
    cs_location_list = [5, 22, 32]
    # cs_location_list = [11, 17, 30, 31, 64]
    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'], 'long': node_data[csid]['long'],'real_lat': node_data[csid]['lat'],
                                 'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                            'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                            'real_long': node_data[csid]['long'], 'diff_node': 0}



    link_traffic = {}
    for l in link_data.keys():

        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(maxspd - int(maxspd * 0.4), maxspd, 288))

    # for l in link_data.keys():
    #     print(l,link_traffic[l])
        # if int(line[1]) in link_traffic:
        #     link_traffic[int(line[1])].append(float(line[2]))
        # else:
        #     link_traffic[int(line[1])] = [float(line[2])]
        # print(line)
    # print('l', linenum)


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy

def network_info_simple_6():

    f = open('data/node_info_6.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(
                                           line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            # print((line[5]), (line[6]))
            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])
        linenum += 1
    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_info_6.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_data = {}

    for line in rdr:
        if linenum == 0:
            a = line
        else:

            link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()

    cs_info = {}
    cs_location_list = [6, 4]
    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'], 'long': node_data[csid]['long'],'real_lat': node_data[csid]['lat'],
                                 'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                            'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                            'real_long': node_data[csid]['long'], 'diff_node': 0}



    link_traffic = {}
    for l in link_data.keys():

        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(maxspd - maxspd * 0.3, maxspd, 288))

    # for l in link_data.keys():
    #     print(l,link_traffic[l])
        # if int(line[1]) in link_traffic:
        #     link_traffic[int(line[1])].append(float(line[2]))
        # else:
        #     link_traffic[int(line[1])] = [float(line[2])]
        # print(line)
    # print('l', linenum)


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy


def network_info_simple_100():

    f = open('data/node_info_100evs.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(
                                           line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            # print((line[5]), (line[6]))
            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])
        linenum += 1
    print('total nodes', linenum - 1)
    f.close()


    f = open('data/link_info_100evs.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_data = {}

    for line in rdr:
        if linenum == 0:
            a = line
        else:

            link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()

    cs_info = {}
    cs_location_list = [32, 37, 62, 67]
    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'], 'long': node_data[csid]['long'],'real_lat': node_data[csid]['lat'],
                                 'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                            'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                            'real_long': node_data[csid]['long'], 'diff_node': 0}



    link_traffic = {}
    for l in link_data.keys():

        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(maxspd - maxspd * 0.4, maxspd, 288))

    # for l in link_data.keys():
    #     print(l,link_traffic[l])
        # if int(line[1]) in link_traffic:
        #     link_traffic[int(line[1])].append(float(line[2]))
        # else:
        #     link_traffic[int(line[1])] = [float(line[2])]
        # print(line)
    # print('l', linenum)


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy

def network_info_simple_25():

    f = open('data/node_info_25evs.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    node_data = {}
    minx = 1000.0
    miny = 1000.0
    maxx = 0.0
    maxy = 0.0
    for line in rdr:
        if linenum == 0:
            a = line
        else:
            node_data[int(line[0])] = {'NODE_ID': int(line[0]), 'NODE_TYPE': int(line[1]), 'NODE_NAME': line[2],
                                       'lat': float(line[6]),
                                       'long': float(
                                           line[5])}  # 'NODE_ID', 'NODE_TYPE', 'NODE_NAME', 'lat'위도(Y), 'long'
            # print((line[5]), (line[6]))
            if minx > float(line[5]):
                minx = float(line[5])
            if miny > float(line[6]):
                miny = float(line[6])
            if maxx < float(line[5]):
                maxx = float(line[5])
            if maxy < float(line[6]):
                maxy = float(line[6])
        linenum += 1
    print('total nodes', linenum - 1)
    f.close()






    f = open('data/link_info_25evs.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    a = 0
    linenum = 0
    link_data = {}

    for line in rdr:
        if linenum == 0:
            a = line
        else:

            link_data[int(line[0])] = {'LINK_ID': int(line[0]), 'F_NODE': int(line[1]), 'T_NODE': int(line[2]),
                                           'MAX_SPD': float(line[11]), 'LENGTH': float(line[15])/1000, 'CUR_SPD': float(
                        0), 'WEIGHT': float(line[15])}

            # 'LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH' (line[0], line[1], line[2], line[11], line[15])
        linenum += 1
    print('total links', linenum-1)
    f.close()

    cs_info = {}
    # cs_location_list = [32, 37, 62, 67]
    cs_location_list = [11, 13, 31, 33]
    for csid in cs_location_list:
        if csid in cs_info.keys():

            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'], 'long': node_data[csid]['long'],'real_lat': node_data[csid]['lat'],
                                 'real_long': node_data[csid]['long'], 'diff_node': 0}
        else:
            cs_info[csid] = {'CS_ID': csid, 'CS_NAME': csid, 'lat': node_data[csid]['lat'],
                            'long': node_data[csid]['long'], 'real_lat': node_data[csid]['lat'],
                            'real_long': node_data[csid]['long'], 'diff_node': 0}



    link_traffic = {}
    for l in link_data.keys():

        maxspd = link_data[l]['MAX_SPD']
        link_traffic[l] = list(np.random.random_integers(maxspd - maxspd * 0.3, maxspd, 288))

    # for l in link_data.keys():
    #     print(l,link_traffic[l])
        # if int(line[1]) in link_traffic:
        #     link_traffic[int(line[1])].append(float(line[2]))
        # else:
        #     link_traffic[int(line[1])] = [float(line[2])]
        # print(line)
    # print('l', linenum)


    return link_data, node_data, link_traffic, cs_info, minx, miny, maxx, maxy

# network_info('data/20191001_5Min_modified.csv')
# network_info_simple()

# network_info('test')