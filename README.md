
configs：保存模型及数据超参数（以yaml格式）；

data: 保存数据集处理相关代码（torch.utils.data.Dataset / torch_geometric.data.InmemoryDataset）；

datasets：保存原始与处理后的数据集；

evaluators：评价指标；

models: 保存模型代码（torch.nn.Module）；

outputs: 模型输出，分为可视化输出及表格输出；

utils：自定义函数包；

rec_run.py：推荐主程序；

visualize.py：可视化主程序；

