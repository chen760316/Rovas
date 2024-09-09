"""
总体步骤分为如下几步：
1、定义命名空间:为RDF uri定义名称空间。
2、RDF转换:convert_to_rdf函数处理DataFrame行和列到RDF三元组的映射。
表URI:为整个表创建一个URI。
行URI:每一行都表示为唯一的URI。
属性URI:将每个属性映射到唯一的URI，并以文字形式添加值。
外键处理:对于外键，定义表之间的关系。
3、外键:该示例包括通过以_id结尾的列名检查外键。可根据特定模式调整此逻辑。
4、序列化:将RDF图序列化为Turtle格式以供输出。
"""

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS

# 定义namespaces
EX = Namespace("http://example.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

# 定义函数将dataframe转为rdf图
def convert_to_rdf(df, table_name):
    g = Graph()

    # 定义表的URI
    table_uri = URIRef(f"http://example.org/{table_name}")

    # 将每一行(元组)映射到唯一的URI并创建RDF三元组
    for index, row in df.iterrows():
        # 为每一行创建特有的URI
        row_uri = URIRef(f"http://example.org/{table_name}/{index}")

        # 为每个属性添加三元组
        for column in df.columns:
            attr_uri = URIRef(f"http://example.org/{table_name}/{column}")
            attr_value = Literal(row[column])
            g.add((row_uri, attr_uri, attr_value))

            # 为每个属性添加标签
            g.add((attr_uri, RDFS.label, Literal(column)))

        # 如果有外键(例如:引用另一个表)
        # 假设列名以“_id”结尾，表示外键
        for column in df.columns:
            if column.endswith('_id'):
                # 定义外键关系
                fk_uri = URIRef(f"http://example.org/{table_name}/{column}")
                fk_target_uri = URIRef(f"http://example.org/another_table/{row[column]}")
                g.add((row_uri, fk_uri, fk_target_uri))

                # 为外键添加一个标签
                g.add((fk_uri, RDFS.label, Literal(column)))
                g.add((fk_uri, RDF.type, EX.ForeignKey))

    return g

file_path = "../UCI_datasets/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
data = pd.read_excel(file_path)

# 将dataframe转为rdf图
rdf_graph = convert_to_rdf(data, 'DryBean')

# 将RDF图序列化为Turtle格式并打印
print(rdf_graph.serialize(format='turtle'))
