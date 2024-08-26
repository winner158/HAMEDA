from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.lecs.state_lecs import StateLECS
from utils.beam_search import beam_search


class LECS(object):
    NAME = 'lecs'

    # L100C10 ELV:3 
    VEHICLE_CAPACITY = [150., 150., 150.]
    # 车辆单位能耗
    Unit_Energy_Consume = [1., 1., 1.]
    # 车辆最大能耗
    Max_Energy_Consume = [15., 15., 15.]
    # 最大服务时间
    MAX_DURATION = [65., 65., 65.]
    # 车辆速度
    SPEED = [1, 1, 1]
    
    

    @staticmethod
    def get_costs(dataset, obj, pi, veh_list, tour_1, tour_2, tour_3):
        # print("\n entering get_cost function!")
        # 求三辆汽车的路径之和，速度都相同
       
        batch_size, graph_size = dataset['demand'].size()
        num_veh = len(LECS.VEHICLE_CAPACITY)

        # # Check that tours are valid, i.e. contain 0 to n -1, [batch_size, num_veh, tour_len]
        sorted_pi = pi.data.sort(1)[0]
        # Sorting it should give all zeros at front and then 1...n
        assert (torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                sorted_pi[:, -graph_size:]
                ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        demand_with_depot = torch.cat(  # [batch_size, graph_size]
            (
                torch.full_like(dataset['demand'][:, :1], 0),  # pickup problem, set depot demand to -capacity
                dataset['demand']
            ),
            1
        )
        # (1)step 1:增加duration
        duration_with_depot = torch.cat(  # [batch_size, graph_size]
            (
                torch.full_like(dataset['duration'][:, :1], 0),  # pickup problem, set depot demand to -capacity
                dataset['duration']
            ),
            1
        )

        # pi: [batch_size, tour_len]
        d = demand_with_depot.gather(1, pi)
        # (1)step 2:增加duration
        d_dur = duration_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0:num_veh])  # batch_size, 3
        # (1)step 3:增加duration
        used_duration = torch.zeros_like(dataset['duration'][:, 0:num_veh])  # batch_size, 3

        # for veh in range(num_veh):  # num_veha
        for i in range(pi.size(-1)):  # tour_len
            # print('d', i, d[0, i])
            used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] += d[:,
                                                                                         i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            used_cap[used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] < 0] = 0
            used_cap[(tour_1[:, i] == 0), 0] = 0
            assert (used_cap[torch.arange(batch_size), 0] <=
                    LECS.VEHICLE_CAPACITY[0] + 1e-5).all(), "Used more than capacity 1"
            used_cap[(tour_2[:, i] == 0), 1] = 0
            assert (used_cap[torch.arange(batch_size), 1] <=
                    LECS.VEHICLE_CAPACITY[1] + 1e-5).all(), "Used more than capacity 2"
            used_cap[(tour_3[:, i] == 0), 2] = 0
            assert (used_cap[torch.arange(batch_size), 2] <=
                    LECS.VEHICLE_CAPACITY[2] + 1e-5).all(), "Used more than capacity 3"
            

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # batch_size, graph_size+1, 2

        # [batch_size, tour_len, 2]
        dis_1 = loc_with_depot.gather(1, tour_1[..., None].expand(*tour_1.size(), loc_with_depot.size(-1)))
        dis_2 = loc_with_depot.gather(1, tour_2[..., None].expand(*tour_2.size(), loc_with_depot.size(-1)))
        dis_3 = loc_with_depot.gather(1, tour_3[..., None].expand(*tour_3.size(), loc_with_depot.size(-1)))

        total_dis_1_Energy = (((dis_1[:, 1:] - dis_1[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_1[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_1[:, -1] - dataset['depot']).norm(p=2, dim=1)) * LECS.Unit_Energy_Consume[0]).unsqueeze(
            -1)  # [batch_size]
        # (2)step 3:增加能耗约束
        # print("\n total_dis_1_Energy:", total_dis_1_Energy)
        # assert (total_dis_1_Energy <=
        #             LECS.Max_Energy_Consume[0] + 1e-5).all(), "Used more than vehicle 1's max energy consume"

        # print("\n total_dis_1 shape:", total_dis_1.shape)
        total_dis_2_Energy = (((dis_2[:, 1:] - dis_2[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_2[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_2[:, -1] - dataset['depot']).norm(p=2, dim=1)) * LECS.Unit_Energy_Consume[1]).unsqueeze(
            -1)  # [batch_size]
        # print("\n total_dis_2_Energy:", total_dis_2_Energy)
        # assert (total_dis_2_Energy <=
        #             LECS.Max_Energy_Consume[1] + 1e-5).all(), "Used more than vehicle 2's max energy consume"
        # total_dis_2_travel_duration = (((dis_2[:, 1:] - dis_2[:, :-1]).norm(p=2, dim=2).sum(1)
        #                 + (dis_2[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
        #                 + (dis_2[:, -1] - dataset['depot']).norm(p=2, dim=1)) / LECS.SPEED[1] )  # [batch_size]
        # print("\n total_dis_2 shape:", total_dis_2.shape)
        total_dis_3_Energy = (((dis_3[:, 1:] - dis_3[:, :-1]).norm(p=2, dim=2).sum(1)
                        + (dis_3[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                        + (dis_3[:, -1] - dataset['depot']).norm(p=2, dim=1)) * LECS.Unit_Energy_Consume[2]).unsqueeze(
            -1)  # [batch_size]
        # print("\n total_dis_3_Energy:", total_dis_3_Energy)
        # assert (total_dis_3_Energy <=
        #             LECS.Max_Energy_Consume[2] + 1e-5).all(), "Used more than vehicle 3's max energy consume"
        # total_dis_3_travel_duration = (((dis_3[:, 1:] - dis_3[:, :-1]).norm(p=2, dim=2).sum(1)
        #                 + (dis_3[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
        #                 + (dis_3[:, -1] - dataset['depot']).norm(p=2, dim=1)) / LECS.SPEED[2])
        
        for i in range(pi.size(-1)):  # tour_len
                  
            # (1)step 3:增加duration(包含车辆运动时间)
            used_duration[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] += d_dur[:, 
                                                                                         i]  # This will reset/make duration negative if i == 0, e.g. depot visited
            used_duration[used_duration[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] < 0] = 0
            used_duration[(tour_1[:, i] == 0), 0] = 0
            # print("\n total_dis_1_travel_duration:", used_duration[torch.arange(batch_size), 2])
            # assert (used_duration[torch.arange(batch_size), 0] <=
            #         LECS.MAX_DURATION[0] + 1e-5).all(), "Used more than vehicle 1's max duration"
            used_duration[(tour_2[:, i] == 0), 1] = 0
            # print("\n total_dis_2_travel_duration:", used_duration[torch.arange(batch_size), 2])
            # assert (used_duration[torch.arange(batch_size), 1] <=
            #         LECS.MAX_DURATION[1] + 1e-5).all(), "Used more than vehicle 2's max duration"
            used_duration[(tour_3[:, i] == 0), 2] = 0
            # print("\n total_dis_3_travel_duration:", used_duration[torch.arange(batch_size), 2])
            # assert (used_duration[torch.arange(batch_size), 2] <=
            #         LECS.MAX_DURATION[2] + 1e-5).all(), "Used more than vehicle 3's max duration"
            # print("\n used_duration shape:", used_duration[torch.arange(batch_size), 2].shape)
            # torch.Size([128])
             
            
        
        # print("\n total_dis_1_duration shape:", total_dis_1_travel_duration.shape)
        # # total_dis_3 shape: torch.Size([128, 1])

        # print("\n total_dis_1_duration:", total_dis_1_travel_duration)
        # print("\n entering total_dis function!")
        total_dis_Energy = torch.cat((total_dis_1_Energy, total_dis_2_Energy, total_dis_3_Energy), -1)
        # print("\n total_dis shape:", total_dis.shape)
        # if obj == 'min-max':
        #     # print("\n entering min-max function!")
        #     return torch.max(total_dis, dim=1)[0], None
        # if obj == 'min-sum':
        #     # print("\n entering min-sum function!")
        return torch.sum(total_dis_Energy, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return LECSDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateLECS.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = LECS.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, duration, max_duration, energy_consume, c_size, distin, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'distin': torch.tensor(distin, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float),
        'duration': torch.tensor(duration, dtype=torch.float),
        'max_duration': torch.tensor(max_duration, dtype=torch.float),
        'energy_consume': torch.tensor(energy_consume, dtype=torch.float),
        'c_size': c_size,
    }


class LECSDataset(Dataset):
    # c_size 是充电站的个数
    def __init__(self, filename=None, size=20, c_size=5, num_samples=10000, offset=0, distribution=None):
        super(LECSDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: [20., 25., 30.],
                20: [20., 25., 30.],
                30: [20., 25., 30.],
                40: [20., 25., 30.],
                50: [20., 25., 30.],
                60: [20., 25., 30.],
                80: [20., 25., 30.],
                100: [20., 25., 30.],
                120: [20., 25., 30.],
            }
            # capa = torch.zeros((size, CAPACITIES[size]))
            loc_t = torch.FloatTensor(size, 2).uniform_(0, 1)
            charge_station = torch.FloatTensor(c_size, 2).uniform_(0, 1)
            loc = torch.cat((charge_station, loc_t))
            # 区分充电节点和普通节点
            distin = torch.cat((torch.ones(c_size, dtype=torch.int64), torch.zeros(size, dtype=torch.int64)))
            # Uniform 1 - 9, scaled by capacities 节点载重
            demand = torch.cat((torch.zeros(c_size), (torch.FloatTensor(size).uniform_(0, 6).int() + 1).float()))
            duration = torch.FloatTensor(c_size + size).uniform_(2, 3)
            energy_consume = torch.zeros(c_size + size)
            self.data = [
                {
                    'loc': loc,
                    'distin': distin,
                    # Uniform 1 - 9, scaled by capacities 节点载重
                    'demand': demand,
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    # 车辆的载重
                    'capacity': torch.Tensor(CAPACITIES[size]),
                    # Uniform 1 - 9, scaled by duration 节点服务时间
                    'duration': duration,
                    'max_duration': torch.Tensor(LECS.MAX_DURATION),
                    'energy_consume': energy_consume,
                    'c_size': c_size,
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data


