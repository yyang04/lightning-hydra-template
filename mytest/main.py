import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # 关键：必须使用 instantiate
    # dept = instantiate(cfg.department)
    # for employee in cfg['department']['employees']:
    #     employee = instantiate(cfg['department']['employees'][employee])
    #     print(employee)

    for name_tuple in cfg.department.employees:
        emp = instantiate(name_tuple.values()[0])          # 此时 emp 是 Employee 对象

    # # 验证 employees 中的对象类型
    # print("Type of first employee:", type(dept.employees[0]))
    # print("Employee name:", dept.employees[0].name)
    #
    # dept.show()


if __name__ == "__main__":
    main()