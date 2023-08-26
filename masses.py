"""
Mass list of aminoacids, protenimodifications, glycan monomers, etc.
Contains also all functions for mass calculation.
Source of mass values from:
http://web.expasy.org/findmod/findmod_masses.html
"""

#我需要知道那些支链也碎的mz，所以我需要知道怎么表明立体结构,以糖所处的位置为糖峰标号，新糖峰标记，比如以前的BY无法标支链碎裂峰，现在以糖来做下标
#pglyco结构转换为图结构
#可以做ficharge到percursor charge，丢水，丢NH3， ？三电荷的碎片有吗，我有看过，那还是保留吧   谱图预测可以快速匹配大量碎片
#对于糖来说，如果对称糖结构，那么不同的来源会有相同的mz，出现在一个峰上,可能只能强度平分
#所有的函数都汇聚在pepfragmass（）函数了，可以只用那个函数
#HCD 碎裂，by离子会把糖也全部丢掉，而且只碎糖的时候也会碎裂多根键 找一下HCD碎裂的诊断离子表
#所有可能的糖结构，需要能做成可以理解的模式。
# 。其实还是需要考虑一下，为什么要做谱图搜索，对于算法而言，糖产生一些细小的结构改变，只是去做一些分析。但是对谱图预测而言，需要重新去预测。那我现在最多只能拿已有的糖去做谱图预测。
#IgG每个位点有的糖就那么几十种，实在不行可以按照那个来。
#糖的峰要慢慢check，注意一下mz
#单糖可以通过多步反应，掉中间那个，可以不预测，只作为诊断离子，HCD模式如何计算碎裂多次
#用ETD模式，CID模式，和HCD模式，依次测试
#可以结合sequence searching 和spectral searching，先匹配预测的mz，再匹配预测的强度
#搜索做断两次+单糖+丢Fuc，预测做丢一次+丢Fuc
#有不同的碎裂模式，不同的碎裂模式要用相应的谱图验证
import re
import copy
import dgl
import torch
import ipdb
# --------------------------- Other masses ----------------------------#
# http://www.sisweb.com/referenc/source/exactmas.htm
MASS = {}
MASS["H2O"] = 18.01057
MASS["NH3"] = 17.02655
MASS["H+"] = 1.00728
MASS["H"] = 1.007825
MASS["C"] = 12.0
MASS["N"] = 14.003074
MASS["O"] = 15.994915
MASS["S"] = 31.972072
MASS["P"] = 30.973763
MASS["Na"] = 22.989770
MASS["K"] = 38.963708
MASS["Mg"] = 23.985045

# --------------------------- Aminoacids ------------------------------#
AMINOACID = {}
AMINOACID["A"] = 71.03711
AMINOACID["R"] = 156.10111
AMINOACID["N"] = 114.04293
AMINOACID["J"] = 114.04293  #带糖的N
AMINOACID["D"] = 115.02694
AMINOACID["C"] = 103.00919
AMINOACID["E"] = 129.04259
AMINOACID["Q"] = 128.05858
AMINOACID["G"] = 57.02146
AMINOACID["H"] = 137.05891
AMINOACID["I"] = 113.08406
AMINOACID["L"] = 113.08406
AMINOACID["K"] = 128.09496
AMINOACID["M"] = 131.04049
AMINOACID["F"] = 147.06841
AMINOACID["P"] = 97.05276
AMINOACID["S"] = 87.03203
AMINOACID["T"] = 101.04768
AMINOACID["W"] = 186.07931
AMINOACID["Y"] = 163.06333
AMINOACID["V"] = 99.06841
# ------------------------------ Glycans ------------------------------#
#Glycan的列表，Monoisotopic mass,库中的所有修饰保持大写，输入的大小写都可以兼容，因为可以转换

GLYCAN = {}
GLYCAN["DHEX"] = 146.0579
GLYCAN["HEX"] = 162.0528
GLYCAN["HEXNAC"] = 203.0794
GLYCAN["NEUAC"] = 291.0954
GLYCAN["NEUGC"] = 307.0903
GLYCAN["FUC"] = 146.0579

# --------------------------- Compositions ----------------------------#
# http://www.webqc.org/aminoacids.php

COMPOSITION = {}
COMPOSITION['A'] = {'C': 3, 'H': 7, 'N': 1, 'O': 2}
COMPOSITION['C'] = {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1}
COMPOSITION['D'] = {'C': 4, 'H': 7, 'N': 1, 'O': 4}
COMPOSITION['E'] = {'C': 5, 'H': 9, 'N': 1, 'O': 4}
COMPOSITION['F'] = {'C': 9, 'H': 11, 'N': 1, 'O': 2}
COMPOSITION['G'] = {'C': 2, 'H': 5, 'N': 1, 'O': 2}
COMPOSITION['H'] = {'C': 6, 'H': 9, 'N': 3, 'O': 2}
COMPOSITION['I'] = {'C': 6, 'H': 13, 'N': 1, 'O': 2}
COMPOSITION['K'] = {'C': 6, 'H': 14, 'N': 2, 'O': 2}
COMPOSITION['L'] = {'C': 6, 'H': 13, 'N': 1, 'O': 2}
COMPOSITION['M'] = {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1}
COMPOSITION['N'] = {'C': 4, 'H': 8, 'N': 2, 'O': 3}
COMPOSITION['J'] = {'C': 4, 'H': 8, 'N': 2, 'O': 3}
COMPOSITION['P'] = {'C': 5, 'H': 9, 'N': 1, 'O': 2}
COMPOSITION['Q'] = {'C': 5, 'H': 10, 'N': 2, 'O': 3}
COMPOSITION['R'] = {'C': 6, 'H': 14, 'N': 4, 'O': 2}
COMPOSITION['S'] = {'C': 3, 'H': 7, 'N': 1, 'O': 3}
COMPOSITION['T'] = {'C': 4, 'H': 9, 'N': 1, 'O': 3}
COMPOSITION['V'] = {'C': 5, 'H': 11, 'N': 1, 'O': 2}
COMPOSITION['W'] = {'C': 11, 'H': 12, 'N': 2, 'O': 2}
COMPOSITION['Y'] = {'C': 9, 'H': 11, 'N': 1, 'O': 3}
COMPOSITION["HEX"] = {'C': 6, 'H': 12, 'N': 0, 'O': 6}
COMPOSITION["DHEX"] = {'C': 6, 'H': 12, 'N': 0, 'O': 5}
COMPOSITION["HEXNAC"] = {'C': 8, 'H': 15, 'N': 1, 'O': 6}
COMPOSITION["NEUAC"] = {'C': 11, 'H': 19, 'N': 1, 'O': 9}
COMPOSITION["NEUGC"] = {'C': 11, 'H': 19, 'N': 1, 'O': 10}
COMPOSITION["H2O"] = {'H': 2, 'O': 1}

# --------------------------- Proteinmodifications---------------------#
PROTEINMODIFICATION = {}
# Cys_CAM Idoacetamide treatment (carbamidation)
PROTEINMODIFICATION["CAR"] = {"mass": 57.021464,
                              "targets": {"C","NTERM"},
                              "composition":{'C':2, 'H':3, 'O':1, 'N':1}
                             }
# Cys_CM, Iodoacetic acid treatment (carboxylation)
PROTEINMODIFICATION["CM"] = {"mass": 58.005479,
                             "targets": {"C"},
                             "composition":{'C': 2, 'H': 2, 'O': 2}
                            }
PROTEINMODIFICATION["OXI"] = {"mass": 15.9949,
                              "targets": {"M"},
                              "composition":{'O': 1}
                             } # MSO
PROTEINMODIFICATION["ACE"] = {"mass": 42.0106,
                               "composition":{'C': 2, 'H': 2, 'O': 1}
                              } # Acetylation
PROTEINMODIFICATION["AMI"] = {"mass": -0.9840,
                               "targets": {"CTERM"},
                               "composition":{'H': 1, 'O': -1, 'N': 1}
                              } # Amidation
PROTEINMODIFICATION["HYD"] = {"mass": 15.9949,
                               "composition":{'O': 1}
                              } # Hydroxylation
PROTEINMODIFICATION["MET"] = {"mass": 14.0157,
                               "composition":{'C': 1, 'H': 2}
                              } # Methylation
PROTEINMODIFICATION["DEH"] = {"mass": -18.0106,
                              "targets": {"S", "T"},
                               "composition":{'H': -2, 'O': -1}
                             } # Dehydration Serine
PROTEINMODIFICATION["DEA"] = {"mass": 0.9840,
                               "targets": {"N", "Q"},
                               "composition":{'H': -1, 'O': 1, 'N': -1}
                              } # Deamidation
PROTEINMODIFICATION["GLY"] = {"mass":0.0 ,
                              "targets": {"N", "S", "T"},
                               "composition":{}
                              }
#按照后修饰和氨基酸对PROTEINMODIFICATION里的后修饰进行排序
def getModificationNames():
    """ Return a sorted list of Modification names.
    Only contains modifications with targets declaration"""
    def sort_names(name):
        if "_" in name:
            amino, mod = name.split("_")
            return mod, amino
        return name, " "
    names = set()
    for key in PROTEINMODIFICATION:
        if "targets" in PROTEINMODIFICATION[key]:
            names.add(key)
    return sorted(names, key=sort_names)
# print("Modifications in list",getModificationNames())

# ------------------------------- Glycan graph ---------------------------#
#表示糖的立体结构，通过DFS计算碎裂一条边以后，剩下的糖部分

node2dict={0:"peptide",1:"HexNAc",2:"Hex",3:"Fuc",4:"NeuAc",5:"NeuGc"}
node2char={"P":"peptide","H" : "Hex", "N" : "HexNAc", "A" : "NeuAc", "G" : "NeuGc" , "F" :"Fuc"}
node2idx={"P":0,"H" : 2, "N" : 1, "A" : 4, "G" : 5 , "F" :3}

def peptide_process(peptide):
    """Create dict for input sequende.

    Input:LSVECAJK_5.Car._6_2_(N(F)(N(H(H)(N)(H(N)))))
    Output:{'sequence': 'LSVECAJK', 'charge': '2', 'modifications': [{'name': '(Car)1', 'amino': 'C', 'position': 4, 'type': 'mod'}, 
    {'name': '(Hex)3(HexNAc)4(Fuc)1', 'amino': 'J', 'position': 6, 'structure': '(N(F)(N(H(H)(N)(H(N)))))', 'type': 'glyco'}]}

    """
    a=peptide.split("_")
    sequence=a[0]
    # print(sequence)
    charge=a[3]
    mod_dict={}
    mod_dictg={}
    mod_list=[]
    if a[1] !="None":
        mod=a[1].rstrip(".").split(".")
        # print(mod)
        for i in range(0,len(mod),2):
            mod_dict=copy.deepcopy(mod_dict)
            mod_dict["name"]="("+mod[i+1]+")1"
            mod_dict["amino"]=sequence[int(mod[i])-1]
            mod_dict["position"]=int(mod[i])-1
            mod_dict["type"]="mod"
            mod_list.append(mod_dict)
    # print("mod_list",mod_list)
    glyco=a[4]
    # print("glyco",glyco)
    position=int(a[2])
    assert sequence.index("J")==position #对于只含有一个J的肽段这应该一直是对的
    # if sequence.count("J")!=1:
    #     print("alert:sequence.count(J)!=1")
    #     raise AssertionError ("It is a multi-glyco peptide")
    composition={"H":str(glyco.count("H")),"N":str(glyco.count("N")),
                    "A":str(glyco.count("A")),"G":str(glyco.count("G")),
                    "F":str(glyco.count("F"))}
    comp_str=""
    for k in list(composition.keys()):
        if composition[k]!="0":
            comp_str+="("+node2char[k]+")"
            comp_str+=str(composition[k])
    mod_dictg["name"]=comp_str
    mod_dictg["amino"]="J"
    mod_dictg["position"]=position
    mod_dictg["structure"]=glyco
    mod_dictg["type"]="glyco"
    mod_list.append(mod_dictg)
    peptide_dict={"sequence":sequence,"charge":charge,"modifications":mod_list}
    # print(peptide)
    # print(peptide_dict)
    return peptide_dict

def glyco_process_0(glyco:str):
    glyco_ind=re.sub("\w","",glyco)
    glyco_cha=re.sub("\W","",glyco)
    # ipdb.set_trace()
    if len(glyco_ind) != 2*len(glyco_cha):
        print("alert:len(glyco_ind) != 2*len(glyco_cha)")
        raise AssertionError
    node1_index=-1
    node2_index=0
    node1="P"
    node_group=[]
    # ipdb.set_trace()
    for i in range(len(glyco_ind)):
        if glyco_ind[i] =="(":
            node2=glyco_cha[node2_index]
            node1_index+=1
            node2_index+=1
            node=[node1_index,node2_index]
            node_group.append(node)
            # ipdb.set_trace()
            # print(node_group)
            # print("(",node1_index)
            # print("(",node2_index)
            node1=node2
            node1_index=node2_index-1
        if glyco_ind[i]==")":
            node1_index-=1
            node1=glyco_cha[node1_index]
            # print(")",node1_index)
        # ipdb.set_trace()
    return node1_index,node2_index,node_group,glyco_cha
def glyco_process(glyco:str):
    glyco_ind=re.sub("\w","",glyco)
    glyco_cha=re.sub("\W","",glyco)
    # ipdb.set_trace()
    if len(glyco_ind) != 2*len(glyco_cha):
        print("alert:len(glyco_ind) != 2*len(glyco_cha)")
        raise AssertionError
    ##双指针,同时需要dict指明str的index和node index的关系

    node1_ptr=0
    node2_ptr=0
    node1=0##多余的起点
    indict={-1:0,}
    node_count=0
    edge_index=[]
    stack=[-2]
    # ipdb.set_trace()
    while node2_ptr < len(glyco):
        if glyco[node2_ptr]=="(":
            # print("(")
            node1_ptr=stack[-1]
            stack.append(node2_ptr)
            
        elif glyco[node2_ptr]==")":
            # print(")")
            node1_ptr=stack.pop()
            
        else:##碰到字母
            # print("add")
            node_count+=1
            indict[node2_ptr]=node_count
            edge_index.append([indict[node1_ptr+1],indict[node2_ptr]])
        # print(node2_ptr)
        # ipdb.set_trace()
        node2_ptr+=1

    # print(edge_index)
    return node1_ptr,node2_ptr,edge_index,glyco_cha
# print(glyco_process("(N(N(H(H)(H(H))))(F))"))
# glyco_process_0("(N(N(H(H)(H(H))))(F))")
# glyco_process_0("(N(F)(N(H(H(N(H)))(H(N(H(G)))))))")
# glyco_process_0("(N(N(H(H(H(H)))(H(H(H(H)))(H(H(H(H))))))))")
# glyco_process_0("(N(N(H(H(N(H(A)))(N(H)))(H(H))))(F))")
# glyco_process("(N(N(H(H(N(H(A)))(N(H)))(H(H))))(F))")
# ipdb.set_trace()

# global diff
# diff=0
# def glycoprocess_diff(glyco):
#     glyco=glyco["H,N,A,G,F"]
#     graphedge_0=glyco_process_0(glyco)[2]
#     graphedge_1=glyco_process(glyco)[2]
#     if graphedge_0!= graphedge_1:
#         # print(glyco)
#         # print(graphedge_1)
#         # print(graphedge_0)
#         global diff
#         diff+=1
#         # print(diff)
#         # ipdb.set_trace()
#     return diff
#         # ipdb.set_trace()
# import pandas as pd
# mouse_gdb=pd.read_csv("/remote-home/yxwang/test/zzb/DeepGlyco/DeepSweet_v1/code/task_processing/AHGF/pGlyco-N-Mouse.gdb",sep="\t")
# mouse_gdb.apply(glycoprocess_diff,axis=1)
# ipdb.set_trace()

def GlycanFrag_struc(glyco_graph):
    edge_index=glyco_graph[2]
    # print("edge_index",edge_index)
    #eg.[[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11]]
    nodef="P"+glyco_graph[3]
    nodef=[node2idx[i] for i in nodef]
    # print("nodef",nodef)
    #eg. nodef [0, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 3]
    g=dgl.graph(edge_index)
    g.ndata["mononer"]=torch.Tensor(nodef).to(int)
    return g

def GlycanFrag(glyco_graph):
    edge_index=glyco_graph[2]
    # print("edge_index",edge_index)
    #eg.[[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11]]
    nodef="P"+glyco_graph[3]
    nodef=[node2idx[i] for i in nodef]
    # print("nodef",nodef)
    #eg. nodef [0, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 3]
    g=dgl.graph(edge_index)
    g.ndata["mononer"]=torch.Tensor(nodef).to(int)
    glycanfrag={}
    for eid in range(len(g.edges()[0])):
        graphe=dgl.remove_edges(g,eid)
        removed_edge=g.edges()[0][eid].item(),g.edges()[1][eid].item()
        #c从这里可以看出，removed_edge是按照eid的顺序从0到len(g.edges()[0])从图上裂掉的。所以removed_edge的顺序就是eid的顺序
        eid_res=list(dgl.dfs_edges_generator(graphe,0))
        noderes=[0]
        graphe_edges=graphe.edges()
        for eid2 in eid_res:
            node1,node2=graphe_edges[0][eid2],graphe_edges[1][eid2]
            noderes.append(node1.item())
            noderes.append(node2.item())
        noderes=list(set(noderes))
        res_monomer=g.ndata["mononer"][noderes].numpy().tolist()
        res_sugar=[node2dict[i] for i in res_monomer if i != 0]
        # print("removed_edge:",removed_edge,"res_index:",res_monomer,"res_sugar:",res_sugar)
        glycanfrag[removed_edge]=res_sugar
    return glycanfrag

def GlycanFrag_HCD(glyco_graph):  
    edge_index=glyco_graph[2]
    # print("edge_index",edge_index)
    nodef="P"+glyco_graph[3]
    nodef=[node2idx[i] for i in nodef]
    # print("nodef",nodef)
    g=dgl.graph(edge_index)
    g.ndata["mononer"]=torch.Tensor(nodef).to(int)
    glycanfrag={}
    # print("g.edges",g.edges())
    for eid in range(len(g.edges()[0])):
        for eid1 in range(eid,len(g.edges()[0])):
            graphe=dgl.remove_edges(g,[eid,eid1])
            removed_edge1=g.edges()[0][eid].item(),g.edges()[1][eid].item()
            removed_edge2=g.edges()[0][eid1].item(),g.edges()[1][eid1].item()
            removed_edge=[removed_edge1,removed_edge2]
                # ------------------- turn to bidirectional -------------------
            u , v = graphe.edges(order = "eid")
            graphe.add_edges(v , u) # bidirect
            # print("removed_edge_list",removed_edge)
            three_splits=[]

            for dfs_initial_node in [removed_edge1[0],removed_edge1[1],removed_edge2[1]]:
                # print("dfs_initial_node:",dfs_initial_node)
                noderes=[dfs_initial_node]
                eid_res=list(dgl.dfs_edges_generator(graphe,dfs_initial_node))
                #很奇怪，这里可以接受0，但是不能接受removed_edge1[0] removed_edge1[1] removed_edge2[2]
                # print("removed_edge1[0]",removed_edge1[0])
                # print("eid_res",eid_res)

                graphe_edges=graphe.edges()
                # print("graphe_edges",graphe_edges)
                for eid2 in eid_res:
                    # print("eid_res_eid2",eid2)
                    node1,node2=graphe_edges[0][eid2],graphe_edges[1][eid2]
                    noderes.append(node1.item())
                    noderes.append(node2.item())
                noderes=list(set(noderes))
                # ipdb.set_trace()
                res_monomer=g.ndata["mononer"][noderes].numpy().tolist()
                res_sugar=[node2dict[i] for i in res_monomer if i != 0]
                three_splits.append(  {"res_index":res_monomer,"res_sugar":res_sugar})
            # print("removed_edge:",removed_edge,"three_splits:", three_splits)
            removed_str=""
            for n in removed_edge:
                for n1 in n:
                    removed_str+=str(n1)
            glycanfrag[removed_str]=three_splits
    return glycanfrag
# import ipdb
# ipdb.set_trace()
# glyco_graph=glyco_process("(N(F)(N(H(H)(H))))")
# glycanfrag=GlycanFrag_HCD(glyco_graph)
# print(glycanfrag)
#对得到的糖碎片计算mz，算好了放到最后一个函数，做整合



# ------------------------------- Functions ---------------------------#

#计算肽段部分质量
def calcPeptideMass(peptide):
    mass = 0
    for s in peptide["sequence"]:
        try:
            mass += AMINOACID[s]
        except KeyError:
            print("Unknown Aminoacid '"+s+"'!")
            raise
    return mass
# print("Peptide",peptide)
# peptidemass=calcPeptideMass(peptide)
# print("Mass for peptide without modifications",peptidemass)

#计算后修饰的总质量
def calcModificationMass(peptide):
    mass = 0
    if peptide["modifications"] is None:
        return mass
    if len(peptide["modifications"]) ==0:
        return mass
    # TODO: Checks for modification consistency
    # a) amount of amino acids must be in sequence
    # b) a given position (except -1) can only be
    for mod in peptide["modifications"]:
        name=mod["name"]
        for part in re.findall(r"\(.+?\)-?\d+", name.upper()):
            monomer, amount = part.split(")")
            monomer = monomer[1:]
            amount = int(amount)
            if monomer in GLYCAN:
                mass += GLYCAN[monomer]*amount
            elif monomer in MASS:
                mass += MASS[monomer]*amount
            elif monomer in PROTEINMODIFICATION:
                mass += PROTEINMODIFICATION[monomer]["mass"]*amount
            else:
                raise Exception("cannot find monomer {} in {}".format(monomer, name))
    return mass
# modificationmass=calcModificationMass(peptide)
# print("mass for modifications",modificationmass)

#算修饰的mz ver:算糖基质量，可以算加和物质
def calcModpepMass(peptide):
    peptidemass = calcPeptideMass(peptide)
    modificationmass = calcModificationMass(peptide)
    ModpepMass=peptidemass+modificationmass
    return ModpepMass
# ModpepMass=calcModpepMass(peptide)
# print("Mass for peptide with modifications",ModpepMass)

def pepfragmass(input,mode,maxcharge=3):
    peptide=peptide_process(input)
    # print("peptide",peptide)
    for modification in peptide["modifications"]:
        name=modification["name"].replace(" ", "")
        if re.match(r"^(\(.+?\)-?\d+)+$", name) == None:
            print("name",name)
            raise Exception(r"""Input string '{}' doesn't follow the
            regex  '^(\(.+?\)-?\d+)+$'
            Please surrond monomers by brackets followed by the amount of
            the monomer, e.g. '(NeuAc)1(H2O)-1(H+)1'""".format(name))
    sequence=peptide["sequence"]
    length=len(sequence)
    FragCharge=list(range(1,min((int(peptide["charge"])+1),maxcharge)))#FragCharge最大值调小了
    b_ions=[]
    y_ions=[]
    B_ions=[]
    Y_ions=[]
    # 计算b/y离子
    #检查一下糖的碎片模式，碎片电荷以及中性丢失的情况
    #M+H-H2O，M+H， M+2H-H2O， M+2H ...
    if "ETD" in mode: #ETD模式意味着，随着位置糖全部保留，然后碎裂肽段
        ModpepMass = calcModpepMass(peptide)+MASS["H2O"]
        for FrgNumC in range(1,len(sequence)):
            Frgmodification=[]
            for modification in peptide["modifications"]:
                if modification["position"]<=FrgNumC-1:
                    Frgmodification.append(modification)
            Fragpepb=sequence[:FrgNumC]
            Fragb={"sequence":Fragpepb,"modifications":Frgmodification}
            #mass = MASS["H2O"] check一下 percursor, b/y ions, B/Y ions的质量
            FragbMass = calcModpepMass(Fragb)
            FragyMass=ModpepMass-FragbMass
            for ficharge in FragCharge:
                for loss_type in ["noloss","loss_H2O","loss_NH3"]:
                    if loss_type=="noloss":
                        FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)
                        bion={FragTypeb:FragbMass/ficharge+MASS["H+"]}
                        b_ions.append(bion)
                        # print(b_ions)
                        FragTypey = "y" + str(len(sequence)-FrgNumC) + "_" + str(ficharge)
                        yion = {FragTypey: FragyMass / ficharge+MASS["H+"]}
                        y_ions.append(yion)
                    if loss_type=="loss_H2O":
                        FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)+"_loss_H2O"
                        bion={FragTypeb:(FragbMass-MASS["H2O"])/ficharge+MASS["H+"]}
                        b_ions.append(bion)
                        # print(b_ions)
                        FragTypey = "y" + str(len(sequence)-FrgNumC) + "_" + str(ficharge)+"_loss_H2O"
                        yion = {FragTypey: (FragyMass-MASS["H2O"]) / ficharge+MASS["H+"]}
                        y_ions.append(yion)
                    if loss_type=="loss_NH3":
                        FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)+"_loss_NH3"
                        bion={FragTypeb:(FragbMass-MASS["NH3"])/ficharge+MASS["H+"]}
                        b_ions.append(bion)
                        # print(b_ions)
                        FragTypey = "y" + str(len(sequence)-FrgNumC) + "_" + str(ficharge)+"_loss_NH3"
                        yion = {FragTypey: (FragyMass-MASS["NH3"]) / ficharge+MASS["H+"]}
                        y_ions.append(yion)
    #计算B/Y ions
    if "HCD_1" in mode: #HCD_1 mode意味着肽段全保留，糖部分碎一刀,另外如果有Fuc可以丢Fuc
        ModpepMass = calcModpepMass(peptide)+MASS["H2O"]
        for i in peptide["modifications"]:
            if i["type"]=="glyco":
                structure=i["structure"]
                glyco_graph=glyco_process(structure)
                # import ipdb
                # ipdb.set_trace() 
                glycanfrag=GlycanFrag(glyco_graph)
                keys=list(glycanfrag.keys())
                values=list(glycanfrag.values())
                # print(values)
                # print(keys)
                peptide_copy=copy.deepcopy(peptide)
                # import ipdb
                # ipdb.set_trace() 
                # print("FragY",peptide)
                # print("FragY2",peptide_copy)
                for k in range(len(keys)):
                    #将k认为是eid的列表，可以考虑是不是对
                    Glycan=values[k]
                    # print("Glycan",Glycan)
                    composition={"H":Glycan.count("Hex"),"N":Glycan.count("HexNAc"),
                    "A":Glycan.count("NeuAc"),"G":Glycan.count("NeuGc"),"F":Glycan.count("Fuc")}
                    comp_str=""
                    for m in list(composition.keys()):
                        if composition[m]!=0:
                            comp_str+="("+node2char[m]+")"
                            comp_str+=str(composition[m])
                    i["name"]=comp_str  #这里改变了peptide,存了peptide的原始备份peptide_copy
                    # print("comp_str",comp_str)
                    FragYMass = calcModpepMass(peptide)+MASS["H2O"]  #h2o需要核实
                    # print(FragYMass)
                    FragBMass=ModpepMass-FragYMass
                    for ficharge in FragCharge:
                        for loss_type in ["noloss","loss_H2O","loss_NH3","loss_FUC"]:
                            if loss_type=="noloss":
                                FragTypeY = "Y" + str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_"+ str(ficharge)
                                Yion = {FragTypeY: round(FragYMass / ficharge+MASS["H+"],5)}
                                Y_ions.append(Yion)
                                FragTypeB= "B"+ str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)
                                Bion = {FragTypeB: round(FragBMass / ficharge+MASS["H+"],5)}
                                B_ions.append(Bion)
                            if loss_type=="loss_H2O":
                                FragTypeY= "Y" + str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)+"_loss_H2O"
                                Yion={FragTypeY: round((FragYMass-MASS["H2O"]) / ficharge+MASS["H+"],5)}
                                Y_ions.append(Yion)
                                FragTypeB = "B"+ str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)+"_loss_H2O"
                                Bion = {FragTypeB: round((FragBMass-MASS["H2O"]) / ficharge+MASS["H+"],5)}
                                B_ions.append(Bion)
                            if loss_type=="loss_NH3":
                                FragTypeY= "Y" + str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)+"_loss_NH3"
                                Yion={FragTypeY: round((FragYMass-MASS["NH3"]) / ficharge+MASS["H+"],5)}
                                Y_ions.append(Yion)
                                FragTypeB = "B"+ str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)+"_loss_NH3"
                                Bion = {FragTypeB: round((FragBMass-MASS["NH3"]) / ficharge+MASS["H+"],5)}
                                B_ions.append(Bion)
                            if len(Glycan)>1 and Glycan[1]== "Fuc" and loss_type=="loss_FUC":
                                FragTypeY= "Y" + str(keys[k][0]) + str(keys[k][1])+"_" + str(k) + "_" + str(ficharge)+"_loss_FUC"
                                Yion={FragTypeY: round((FragYMass-GLYCAN["FUC"]) / ficharge+MASS["H+"],5)}
                                Y_ions.append(Yion)
        peptide=peptide_copy
        # print(Y_ions)
    if "HCD_by" in mode:
        #HCD的b,y碎片需要脱糖，或者有一个HexNac
        copy_num=0
        for gly_HCD in ["(HexNAc)0","(HexNAc)1"]:
            for modification in peptide["modifications"]:
                    if modification["type"]=="glyco":
                        if copy_num==0:
                            glyco_mod=copy.deepcopy(modification["name"])
                            glysite=modification["position"]
                            # print("glyco_mod",glyco_mod)
                            copy_num+=1
                        # assert modification["structure"][1] =="N"
                        modification["name"]=gly_HCD
                        glyco_aftermod=copy.deepcopy(modification)
                        ModpepMass = calcModpepMass(peptide)+MASS["H2O"]
            for FrgNumC in range(1,len(sequence)):
                Frgmodification=[]
                for modification in peptide["modifications"]:
                    if modification["position"]<=FrgNumC-1:
                            Frgmodification.append(modification)
                Fragpepb=sequence[:FrgNumC]
                Fragb={"sequence":Fragpepb,"modifications":Frgmodification}
                #mass = MASS["H2O"] check一下 percursor, b/y ions, B/Y ions的质量
                FragbMass = calcModpepMass(Fragb)
                FragyMass=ModpepMass-FragbMass
                for ficharge in FragCharge:
                    for loss_type in ["noloss","loss_H2O","loss_NH3"]:
                        if loss_type=="noloss":
                            FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)+"_"+gly_HCD
                            if FragTypeb.startswith("b") and int(FrgNumC)<=int(glysite) and gly_HCD=="(HexNAc)1":
                                pass
                            else:
                                bion={FragTypeb:round(FragbMass/ficharge+MASS["H+"],5)}
                                b_ions.append(bion)
                            # print(b_ions)
                            # import ipdb
                            # ipdb.set_trace()
                            FragTypey = "y" + str(length-FrgNumC) + "_" + str(ficharge)+"_"+gly_HCD
                            if FragTypey.startswith("y") and FrgNumC>int(glysite) and gly_HCD=="(HexNAc)1":
                                 pass
                            else:
                                yion = {FragTypey: round(FragyMass / ficharge+MASS["H+"],5)}
                                y_ions.append(yion)
                        if loss_type=="loss_H2O":
                            FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)+"_loss_H2O_"+gly_HCD
                            if FragTypeb.startswith("b") and int(FrgNumC)<=int(glysite) and gly_HCD=="(HexNAc)1":
                                pass
                            else:
                                bion={FragTypeb:round((FragbMass-MASS["H2O"])/ficharge+MASS["H+"],5)}
                                b_ions.append(bion)
                            # print(b_ions)
                            FragTypey = "y" + str(len(sequence)-FrgNumC) + "_" + str(ficharge)+"_loss_H2O_"+gly_HCD
                            if FragTypey.startswith("y") and FrgNumC>int(glysite) and gly_HCD=="(HexNAc)1":
                                 pass
                            else:
                                yion = {FragTypey: round((FragyMass-MASS["H2O"]) / ficharge+MASS["H+"],5)}
                                y_ions.append(yion)
                        if loss_type=="loss_NH3":
                            FragTypeb= "b"+str(FrgNumC)+"_"+str(ficharge)+"_loss_NH3_"+gly_HCD
                            if FragTypeb.startswith("b") and int(FrgNumC)<=int(glysite) and gly_HCD=="(HexNAc)1":
                                pass
                            else:
                                bion={FragTypeb:round((FragbMass-MASS["NH3"])/ficharge+MASS["H+"],5)}
                                b_ions.append(bion)
                            # print(b_ions)
                            FragTypey = "y" + str(len(sequence)-FrgNumC) + "_" + str(ficharge)+"_loss_NH3_"+gly_HCD
                            if FragTypey.startswith("y") and FrgNumC>int(glysite) and gly_HCD=="(HexNAc)1":
                                 pass
                            else:
                                yion = {FragTypey: round((FragyMass-MASS["NH3"]) / ficharge+MASS["H+"],5)}
                                y_ions.append(yion)
        #HCD_1 mode意味着肽段全保留，糖部分碎一刀,另外如果有Fuc可以丢Fuc
        #B/Y ions糖可以碎裂两刀或者一刀,虽然可以碎三次以上，但是碎太多，搜索空间过大，而且不具有结构区分性
        #另外加上单糖
        #有很多m/z重复的，需要去重
    if "HCD_BY_2" in mode:
        ModpepMass = calcModpepMass(peptide)+MASS["H2O"]
        for i in peptide["modifications"]:
            if i["type"]=="glyco":
                structure=i["structure"]
                # ipdb.set_trace()
                p = re.compile(r'[()](.*?)[)]', re.S)
                # ipdb.set_trace()
                glycan_all=re.findall(p,i["name"])
                # print("glycan_all",glycan_all)
                # ipdb.set_trace()
                # FragCharge=list(range(1,min((int(peptide["charge"])+1),4)))
                for mono_sugar in glycan_all:
                    for ficharge in FragCharge:
                        for addition_type in ["","_add_H2O","_loss_H2O"]:
                            FragTypeB = "B_" + str(ficharge)+"_"+mono_sugar+addition_type
                            if addition_type =="":
                                FragBMass=GLYCAN[mono_sugar.upper()]
                            if addition_type =="_add_H2O":
                                FragBMass=GLYCAN[mono_sugar.upper()]+MASS["H2O"]
                            if addition_type =="_loss_H2O":
                                FragBMass=GLYCAN[mono_sugar.upper()]-MASS["H2O"]
                            Bion = {FragTypeB: FragBMass / ficharge+MASS["H+"]}
                            B_ions.append(Bion)
                # ipdb.set_trace()
                glyco_graph=glyco_process(structure)
                glycanfrag=GlycanFrag_HCD(glyco_graph)
                keys=list(glycanfrag.keys())
                values=list(glycanfrag.values())
                # print(values)
                # print(keys)
                # ipdb.set_trace()
                peptide_copy=copy.deepcopy(peptide)
                for k in range(len(keys)):
                    Glycan=values[k]
                    # print("Glycan",Glycan)
                    # print("bond",keys[k])
                    # ipdb.set_trace()
                    frg_num=0
                    for res in Glycan:
                        res_index=res["res_index"]
                        if 0 in res_index: #Y_ions
                            frg_num+=1
                            FragNumY=keys[k]+"_"+str(frg_num)
                            # print("FragNumY",FragNumY)
                            res_sugar=res["res_sugar"]
                            # print("res_sugar",res_sugar)
                            composition={"H":res_sugar.count("Hex"),"N":res_sugar.count("HexNAc"),
                            "A":res_sugar.count("NeuAc"),"G":res_sugar.count("NeuGc"),"F":res_sugar.count("Fuc")}
                            comp_str=""
                            for m in list(composition.keys()):
                                if composition[m]!=0:
                                    comp_str+="("+node2char[m]+")"
                                    comp_str+=str(composition[m])
                            i["name"]=comp_str  #这里改变了peptide,存了peptide的原始备份peptide_copy
                            # ipdb.set_trace()
                            # print("comp_str",comp_str)
                            FragYMass = calcModpepMass(peptide)+MASS["H2O"]  #h2o需要核实
                            # print(FragYMass)
                            for ficharge in FragCharge:
                                for loss_type in ["noloss","loss_H2O","loss_NH3","loss_FUC"]:
                                    if loss_type=="noloss":
                                        FragTypeY = "Y" + FragNumY+"_" + str(ficharge)
                                        Yion = {FragTypeY: round(FragYMass / ficharge+MASS["H+"],5)}
                                        Y_ions.append(Yion)
                                    if loss_type=="loss_H2O":
                                        FragTypeY= "Y" + FragNumY+"_" + str(ficharge)+"_loss_H2O"
                                        Yion={FragTypeY: round((FragYMass-MASS["H2O"]) / ficharge+MASS["H+"],5)}
                                        Y_ions.append(Yion)
                                    if loss_type=="loss_NH3":
                                        FragTypeY= "Y" + FragNumY+"_" + str(ficharge)+"_loss_NH3"
                                        Yion={FragTypeY: round((FragYMass-MASS["NH3"]) / ficharge+MASS["H+"],5)}
                                        Y_ions.append(Yion)
                                    if len(res_sugar)>1 and res_sugar[1]== "Fuc" and loss_type=="loss_FUC":
                                        FragTypeY= "Y" + FragNumY+"_" + str(ficharge) + "_" + str(ficharge)+"_loss_FUC"
                                        Yion={FragTypeY: round((FragYMass-GLYCAN["FUC"]) / ficharge+MASS["H+"],5)}
                                        Y_ions.append(Yion)
                        else:
                            frg_num+=1
                            FragNumB=keys[k]+"_"+str(frg_num)
                            # print(FragNumB)
                            res_sugar=res["res_sugar"]    
                            composition={"H":res_sugar.count("Hex"),"N":res_sugar.count("HexNAc"),
                            "A":res_sugar.count("NeuAc"),"G":res_sugar.count("NeuGc"),"F":res_sugar.count("Fuc")}
                            FragBMass=0
                            for m in list(composition.keys()):
                                if composition[m]!=0:
                                    # print(node2char[m].upper())
                                    FragBMass+=GLYCAN[node2char[m].upper()]*composition[m]
                            # print("FragBMass",FragBMass)
                            for ficharge in FragCharge:
                                for loss_type in ["noloss","loss_H2O","loss_NH3","loss_FUC"]:
                                    if loss_type=="noloss":
                                        FragTypeB= "B"+ FragNumB+"_" + str(ficharge)
                                        Bion = {FragTypeB: round(FragBMass / ficharge+MASS["H+"],5)}
                                        B_ions.append(Bion)
                                    if loss_type=="loss_H2O":
                                        FragTypeB = "B"+ FragNumB+"_" + str(ficharge)+"_loss_H2O"
                                        Bion = {FragTypeB: round((FragBMass-MASS["H2O"]) / ficharge+MASS["H+"],5)}
                                        B_ions.append(Bion)
                                    if loss_type=="loss_NH3":
                                        if "HexNAc" in res_sugar or "NEUAC" in res_sugar or "NEUGC" in res_sugar:
                                            # print("res_sugar",res_sugar)    
                                            FragTypeB = "B"+ FragNumB+"_" + str(ficharge)+"_loss_NH3"
                                            Bion = {FragTypeB: round((FragBMass-MASS["NH3"]) / ficharge+MASS["H+"],5)}
                                            B_ions.append(Bion)
                                    if len(res_sugar)>1 and res_sugar[1]== "Fuc" and loss_type=="loss_FUC":
                                            # print("res_sugar",res_sugar)    
                                        FragTypeB = "B"+ FragNumB+"_" + str(ficharge)+"_loss_FUC"
                                        Bion = {FragTypeB: round((FragBMass-GLYCAN["FUC"]) / ficharge+MASS["H+"],5)}
                                        B_ions.append(Bion)
        peptide=peptide_copy
        # print(Y_ions)
    return b_ions,y_ions,B_ions,Y_ions
# input="AEAVGETLTLPGLVSADJGTYTCEAANK_23.Car._63002_3_(N(F)(N(H(H(H)(H))(H(N(H)(F))))))"
# input="EDGMLPAJR_None_7_3_(N(F)(N(H(H(H))(H))))"
# input="GGJGTICDNQR_7.Car._2_2_(N(F)(N(H(H(N))(H(N)(N)))))"
# mz_calc=pepfragmass(input,["HCD_by"])
# print(mz_calc)
# import ipdb
# ipdb.set_trace()

# print("Fragment MZ for modifided peptides",pepfragmass(peptide))