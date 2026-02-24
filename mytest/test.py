from omegaconf import OmegaConf

data = {
    'name': 'myTest',
    'employees': [
        {'name': 'Alice', 'salary': 120},
        {'name': 'Bob', 'salary': 122}
    ]
}

# 保存为 YAML
OmegaConf.save(data, 'output.yaml')
cfg = OmegaConf.load('output.yaml')
print(cfg.name)               # 输出: myTest
print(cfg.employees[0].name)  # 输出: Alice
print(cfg.employees[1].salary) # 输出: 122