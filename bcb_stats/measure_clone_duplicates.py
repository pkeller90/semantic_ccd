import psycopg2
import pandas as pd
import pandas.io.sql as pdsql
import os

#TODO change password of user sa to sa (empty by default but no empty password allowed by psycopg2)
conn = psycopg2.connect("dbname=bcb user='sa' password='sa' host='localhost' port=5435")


# Fetch all clone pairs
query = 'SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, FUNCTIONALITY_ID, SYNTACTIC_TYPE, SIMILARITY_LINE FROM CLONES'
# WHERE SYNTACTIC_TYPE=1;'   
clone_pairs = pdsql.read_sql_query(query, conn)
clone_pairs = clone_pairs.astype({"function_id_one": int, "function_id_two": int, "syntactic_type": int,
                                  "similarity_line": float})

def compute_type(row):
    if row["syntactic_type"] == 3:
        if row["similarity_line"] < 0.5:
            return int(4)
        else:
            return int(3)
    else:
        return int(row["syntactic_type"])


clone_pairs["type"] = clone_pairs.apply(compute_type, axis=1).astype(int)

print(clone_pairs.head())

print("Done FETCHING data from DB, can close now")


def get_unique_code_ids(ds):
    code_ids_by_ct = [None,set(),set(),set(),set()]
    
    for _, r in ds.iterrows():
        id1 = r["function_id_one"]
        id2 = r["function_id_two"]
        ctype = int(r["type"])
        code_ids_by_ct[ctype].add(id1)
        code_ids_by_ct[ctype].add(id2)
    for i in range(1, 5):
        print("T-%s: %s different snippets" % (i, len(code_ids_by_ct[i])))
    print("T4 - T1 snippets: %s different snippets" % len(code_ids_by_ct[4]-code_ids_by_ct[1]))
    print("T4 - (T1+T2) snippets: %s different snippets" % len((code_ids_by_ct[4]-code_ids_by_ct[2])-code_ids_by_ct[1]))
    print("T4 - (T1+T2+T3) snippets: %s different snippets" % len(((code_ids_by_ct[4]-code_ids_by_ct[3])-code_ids_by_ct[2])-code_ids_by_ct[1]))
    return code_ids_by_ct
cids_by_ct = get_unique_code_ids(clone_pairs)


# In[13]:


for i in range(1, 5):
    print("Number of T-%s clone pairs: %s" % (i, len(clone_pairs[clone_pairs["type"] == i].index)))

def filter_blacklisted_pairs(ds, blacklist):
    ds = ds[(~ds["function_id_one"].isin(blacklist)) &
            (~ds["function_id_two"].isin(blacklist))]
    return ds

for i in range(1, 5):
    pairs = filter_blacklisted_pairs(clone_pairs[clone_pairs["type"] == i], cids_by_ct[1]) if i != 1 else clone_pairs[clone_pairs["type"] == i]
    print("Number of T-%s clone pairs: %s" % (i, len(pairs.index)))

t1_classes = {}
seen_ids = set()
class_sets = []

for _, r in clone_pairs[clone_pairs["type"] == 1].iterrows():
    id1 = str(int(r["function_id_one"]))
    id2 = str(int(r["function_id_two"]))
    class_set = None
    if id1 in seen_ids:
        class_set = t1_classes[id1]
    if id2 in seen_ids:
        class_set = t1_classes[id2]
    if class_set is None:
        class_set = set()
        class_sets.append(class_set)
    class_set.add(id1)
    class_set.add(id2)
    seen_ids.add(id1)
    seen_ids.add(id2)
    t1_classes[id1] = class_set
    t1_classes[id2] = class_set

    
class_representants = {next(iter(x)): x for x in class_sets}
class_representant_mapping = {}
for k, v in class_representants.items():
    for l in v:
        class_representant_mapping[l] = k

print("mappings created")

def map_to_representant(r):
    id1 = str(int(r["function_id_one"]))
    id2 = str(int(r["function_id_two"]))
    mid1 = class_representant_mapping.get(id1, default=id1)
    mid2 = class_representant_mapping.get(id2, default=id2)
    nid1 = int(mid1)
    nid2 = int(mid2)
    if nid1 > nid2:
        nid1, nid2 = nid2, nid1
    return nid1, nid2
    
#test = clone_pairs.apply(map_to_representant, axis=1)
#test.head(5)
#clone_pairs.head(5)


# In[ ]:


for i in range(1, 5):
    cp_by_type = (clone_pairs[clone_pairs["type"] == i])[["function_id_one", "function_id_two"]].clone()
    print(cp_by_type.head(5))
    tmp = cp_by_type.apply(map_to_representant, axis=1)
    tmp.columns = ["nid1", "nid2"]
    uniques = len(tmp.groupby(['nid1', 'nid2']).groups)
    print("Number of unique T-%s clone pairs: %s" % (i, uniques))
