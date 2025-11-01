import pandas as pd

class InputClassifier:
    def __init__(self, filepath):
        """
        初始化分类器：从 Excel 文件中读取所有组合 (名称+途径) -> 分类
        """
        self.filepath = filepath  # Store filepath for later use
        self.class_map = {}  # (name, way) -> 分类
        self.categories = {
                '1': '晶体',
                '2': '胶体',
                '3': '水',
                '4': '口服',
                '5': '其他',
                '6': '未知(不保存)',
        }

        self.type_mapping = {
            "晶体": 0,  # 晶体
            "胶体": 1,  # 胶体
            "葡萄糖": 2  # 葡萄糖
        }

        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            print("sucesss load excel")
        except FileNotFoundError:
            # 创建一个空的 Excel 文件
            df = pd.DataFrame(columns=['入量:名称', '入量:途径', '分类'])
            df.to_excel(filepath, index=False, engine='openpyxl')

        # 必须包含以下三列
        required_columns = ['入量:名称', '入量:途径', '分类']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少列: {col}")

        for _, row in df.iterrows():
            name = str(row['入量:名称']).strip()
            way = str(row['入量:途径']).strip()
            category = str(row['分类']).strip()
            print(f"加载分类: {name}, {way} -> {category}")
            self.class_map[(name, way)] = category

    def query(self, name, way):
        """
        查询 (name, way) 对应的分类；如果找不到则人工判断并添加
        """
        if way is None:
            way = 'nan'
        if name is None:
            name = 'nan'

        key = (str(name).strip(), str(way).strip())
        category = self.class_map.get(key, None)
        
        if category is None:
            # 人工判断分类
            print(f"\n请为以下条目分类:")
            print(f"名称: {key[0]}")
            print(f"途径: {key[1]}")
            while True:
                print("\n请选择分类:")
                for num, cat in self.categories.items():
                    print(f"{num}. {cat}")
                try:
                    choice = input("请输入数字: ").strip()
                    if choice in self.categories:
                        category = self.categories[choice]
                        break
                    else:
                        print("无效输入，请输入有效的数字")
                except KeyboardInterrupt:
                    raise
                except:
                    print("输入错误，请重试")
            
            # 添加新的分类
            if category is not None and category != "未知":
                self.add(key[0], key[1], category)
        return category

    def add(self, name, way, category):
        """
        添加新的 (name, way) -> 分类 映射
        """
        if self.class_map.get((name, way)) is not None:
            # print(f"已存在分类: {name}, {way} -> {self.class_map[(name, way)]}")
            return
        key = (str(name).strip(), str(way).strip())
        self.class_map[key] = str(category).strip()

    def save(self, output_path=None):
        """
        保存分类结果到Excel文件
        """
        if output_path is None:
            output_path = self.filepath
            
        data = []
        for (name, way), category in self.class_map.items():
            data.append({
                '入量:名称': name,
                '入量:途径': way,
                '分类': category
            })
            
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)

    def __len__(self):
        return len(self.class_map)

