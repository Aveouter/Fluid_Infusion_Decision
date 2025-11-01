def project_rate_values(preds, rate_type='crystal'):
    """
    将预测值（单个或列表）投射到预定义的速度列表中最接近的合法值。

    参数:
        preds (float or list of floats): 预测值或预测值列表
        rate_type (str): 'crystal', 'colloid', or 'water'

    返回:
        float 或 list: 投射后的值或值列表
    """
    # 定义速度值列表
    CRYSTAL = sorted([
        15, 25, 35, 45, 75, 90, 10, 30, 140, 160, 170, 175, 220, 210, 125,
        650, 550, 1200, 1100, 270, 280, 275, 1000, 1450, 750, 85, 110, 1600
    ])
    COLLOID = sorted([
        0, 15, 20, 25, 30, 35, 40, 50, 60, 70, 75, 80, 85, 90, 100,
        110, 120, 125, 130, 140, 150, 160, 170, 175, 180, 190, 200,
        210, 220, 240, 250, 270, 280, 300, 330, 350, 400, 450, 500,
        550, 600, 650, 700, 750, 800, 1000
    ])
    WATER = sorted([
        0, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 85, 90, 100,
        110, 120, 125, 130, 140, 150, 160, 170, 175, 180, 200, 210, 220,
        225, 240, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        800, 900, 1000, 1100, 1200, 1600
    ])
    
    # 获取对应速度列表
    rate_map = {
        'crystal': CRYSTAL,
        'colloid': COLLOID,
        'water': WATER
    }
    types = [ 'crystal', 'colloid', 'water' ]
    if rate_type not in rate_map:
        raise ValueError(f"Invalid rate_type: {rate_type}. Must be 'crystal', 'colloid', or 'water'.")

    # 定义投射函数
    def project_single(p):
        return min(target_list, key=lambda x: abs(x - p))

    # 处理单个或列表
    if isinstance(preds, (int, float)):
        target_list = rate_map[rate_type]
        return project_single(preds)
    elif isinstance(preds, list):
        res = []
        for _,p in enumerate(preds):
            target_list = rate_map[types[_]]
            res.append(project_single(p))
        return res
    else:
        raise TypeError("Input must be a float, int, or list of floats.")
