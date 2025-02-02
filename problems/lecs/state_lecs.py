import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np
import copy


class StateLECS(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc, [batch_size, graph_size+1, 2]
    demand: torch.Tensor
    # capacity: torch.Tensor
    duration: torch.Tensor
    energy_consume: torch.Tensor
    # 新增
    distin: torch.Tensor
    c_size: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    veh: torch.Tensor  # numver of vehicles

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    used_duration: torch.Tensor
    used_energy_consume: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    # segment_length: torch.Tensor

    # L100C10 ELV:3 
       # L100C10 ELV:3 
    VEHICLE_CAPACITY = [150., 150., 150.]
    # 车辆单位能耗
    Unit_Energy_Consume = [1., 1., 1.]
    # 车辆最大能耗
    Max_Energy_Consume = [15., 15., 15.]
    # 最大服务时间
    MAX_DURATION = [65., 65., 65.]
    # 车辆速度

    # 车辆速度
    SPEED = [1, 1, 1]
   
     # 车辆单位能耗
    # Unit_Energy_Consume = [1., 1., 1.]

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return self.visited_[:, None, :].expand(self.visited_.size(0), 1, -1).type(torch.ByteTensor)

    @property
    def dist(self):  # coords: []
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                veh=self.veh[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                used_duration=self.used_duration[key],
                used_energy_consume=self.used_energy_consume[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateLECS, self).__getitem__(key)


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        duration = input['duration']
        energy_consume = input['energy_consume']
        distin = input['distin']
        c_size = int(input['c_size'][-1])

        # print("\n c_size shape:", c_size)

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        # n_loc -= c_size
        # print("\n n_loc :", n_loc)
        return StateLECS(
            coords=torch.cat((depot[:, None, :], loc), -2),  # [batch_size, graph_size, 2]]
            demand=torch.cat((torch.zeros(batch_size, 1, device=loc.device), demand), 1),
            distin=distin,
            c_size=c_size,
            duration=torch.cat((torch.zeros(batch_size, 1, device=loc.device), duration), 1),
            energy_consume=torch.cat((torch.zeros(batch_size, 1, device=loc.device), energy_consume), 1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            veh=torch.arange(len(StateLECS.VEHICLE_CAPACITY), dtype=torch.int64, device=loc.device)[:, None],
            # prev_a is current node
            prev_a=torch.zeros(batch_size, len(StateLECS.VEHICLE_CAPACITY), dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, len(StateLECS.VEHICLE_CAPACITY)),
            used_duration=duration.new_zeros(batch_size, len(StateLECS.MAX_DURATION)),
            used_energy_consume=demand.new_zeros(batch_size, len(StateLECS.Max_Energy_Consume)),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc - c_size + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, len(StateLECS.VEHICLE_CAPACITY), device=loc.device),
            cur_coord=input['depot'][:, None, :].expand(batch_size, len(StateLECS.VEHICLE_CAPACITY), -1),
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    # def get_final_cost(self):
    #     assert self.all_finished()
    #     # coords: [batch_size, graph_size+1, 2]
    #     return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, veh):  # [batch_size, num_veh]

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        prev_a = selected  # [batch_size, num_veh]
        batch_size, _ = selected.size()

        # Add the length, coords:[batch_size, graph_size, 2]
        cur_coord = self.coords.gather(  # [batch_size, num_veh, 2]
            1,
            selected[:, :, None].expand(selected.size(0), len(self.VEHICLE_CAPACITY), self.coords.size(-1))
        )

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        segment_length = (cur_coord[torch.arange(batch_size), veh, :] - self.cur_coord[torch.arange(batch_size), veh, :]).norm(p=2, dim=-1)
        # segment_length shape: torch.Size([128])
        # print("\n segment_length:", segment_length)
        

        # demand:[batch_size, n_loc+1]
        selected_demand_broad = self.demand[:, :, None].gather(  # [batch_size, num_veh]
            1,
            prev_a[torch.arange(batch_size), veh][:, None, None].expand(prev_a.size(0), len(self.VEHICLE_CAPACITY),
                                                                        self.demand[:, :, None].size(-1))
        ).squeeze(2)
        selected_demand = torch.zeros_like(selected_demand_broad)
        selected_demand[torch.arange(batch_size), veh] = selected_demand_broad[torch.arange(batch_size), veh].clone()

        used_capacity = self.used_capacity
        used_capacity[torch.arange(batch_size), veh] = (self.used_capacity[torch.arange(batch_size), veh] +
                                                        selected_demand[torch.arange(batch_size), veh]) * (
                                                               prev_a[torch.arange(
                                                                   batch_size), veh] != 0).float()  # [batch_size, num)_veh]
        # =============================(1)step 1:增加duration================================
        # duration:[batch_size, n_loc+1]
        selected_duration_broad = self.duration[:, :, None].gather(  # [batch_size, num_veh]
            1,
            prev_a[torch.arange(batch_size), veh][:, None, None].expand(prev_a.size(0), len(self.MAX_DURATION),
                                                                        self.duration[:, :, None].size(-1))
        ).squeeze(2)
        selected_duration = torch.zeros_like(selected_duration_broad)
        selected_duration[torch.arange(batch_size), veh] = selected_duration_broad[torch.arange(batch_size), veh].clone()
        # print("\n  selected_duration[torch.arange(batch_size), veh] shape:",  selected_duration[torch.arange(batch_size), veh])
        used_duration = self.used_duration
        used_duration[torch.arange(batch_size), veh] = (self.used_duration[torch.arange(batch_size), veh] +
                                                        selected_duration[torch.arange(batch_size), veh] + segment_length) * (
                                                               prev_a[torch.arange(
                                                                   batch_size), veh] != 0).float()  # [batch_size, num)_veh]
        # =============================(2)step 1:增加Max_Energy_Consume================================
        # print("\n veh shape:", veh.shape)
        # print("\n veh:", veh)
        
        
        Unit_Energy_Consume_veh = torch.tensor([StateLECS.Unit_Energy_Consume[i] for i in veh]).cuda()
        # print("\n Unit_Energy_Consume_broad shape:", Unit_Energy_Consume_broad.shape)
        # print("\n Unit_Energy_Consume_broad:", Unit_Energy_Consume_broad)

        current_seg_Energy_Consume = segment_length*Unit_Energy_Consume_veh
        # print("\n current_seg_Energy_Consume shape:", current_seg_Energy_Consume.shape)
        # print("\n current_seg_Energy_Consume:", current_seg_Energy_Consume)
        selected_energy_consume_broad = self.energy_consume[:, :, None].gather(  # [batch_size, num_veh]
            1,
            prev_a[torch.arange(batch_size), veh][:, None, None].expand(prev_a.size(0), len(self.Max_Energy_Consume),
                                                                        self.energy_consume[:, :, None].size(-1))
        ).squeeze(2)
        selected_energy_consume = torch.zeros_like(selected_energy_consume_broad)
        selected_energy_consume[torch.arange(batch_size), veh] = selected_energy_consume_broad[torch.arange(batch_size), veh].clone()
        # StateLECS.Unit_Energy_Consume[veh.numpy()[0]] energy_consume

        used_energy_consume = self.used_energy_consume
        used_energy_consume[torch.arange(batch_size), veh] = (self.used_energy_consume[torch.arange(batch_size), veh] +
                                                              selected_energy_consume[torch.arange(batch_size), veh] + current_seg_Energy_Consume) * (
                                                               prev_a[torch.arange(
                                                                   batch_size), veh] != 0).float()

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[torch.arange(batch_size), veh][:, None, None].expand_as(
                self.visited_[:, :, 0:1]), 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a[torch.arange(batch_size), veh])

    
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, used_duration=used_duration, visited_=visited_,
            used_energy_consume=used_energy_consume, lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        # return self.visited.all()
        # 修改
        return (self.prev_a == 0).all() and self.visited_[:, :, self.c_size:].all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    # need to be modified
    def get_mask(self, veh):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        batch_size = self.visited_.size(0)
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, n_loc]
        else:
            visited_loc = self.visited_[:, 1:][:, None, :]  # [batch_size, 1, n_loc]

        exceeds_cap = (self.demand[self.ids, 1:] + (self.used_capacity[torch.arange(batch_size), veh].unsqueeze(-1))[
            ..., None].expand_as(self.demand[self.ids, 1:]) >
                       ((torch.tensor(self.VEHICLE_CAPACITY)[None, :].expand(batch_size,
                                                                             len(self.VEHICLE_CAPACITY)).cuda())[
                            self.ids, veh][0].unsqueeze(-1))[..., None].expand_as(
                           self.demand[self.ids, 1:]))  
        
        # =============================(1)step 2:增加duration================================
 
        exceeds_dur = (self.duration[self.ids, 1:] + (self.used_duration[torch.arange(batch_size), veh].unsqueeze(-1))[
            ..., None].expand_as(self.duration[self.ids, 1:]) >
                       ((torch.tensor(StateLECS.MAX_DURATION)[None, :].expand(batch_size,
                                                                             len(self.MAX_DURATION)).cuda())[
                            self.ids, veh][0].unsqueeze(-1))[..., None].expand_as(
                           self.duration[self.ids, 1:])) 
        
        # =============================(2)step 2:增加Max_Energy_Consume================================

        exceeds_energy =  (self.energy_consume[self.ids, 1:] + (self.used_energy_consume[torch.arange(batch_size), veh].unsqueeze(-1))[
            ..., None].expand_as(self.energy_consume[self.ids, 1:]) >
                       ((torch.tensor(self.Max_Energy_Consume)[None, :].expand(batch_size,
                                                                             len(self.Max_Energy_Consume)).cuda())[
                            self.ids, veh][0].unsqueeze(-1))[..., None].expand_as(
                           self.energy_consume[self.ids, 1:])) 

        # Nodes that cannot be visited are already visited or too much demand to be served now
        # mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap  # [batch_size, 1, n_loc]
        # print("\n visited_loc.to(exceeds_cap.dtype):", visited_loc.to(exceeds_cap.dtype).shape)
        # print("\n exceeds_cap:", exceeds_cap.shape)
        # print("\n exceeds_dur:", exceeds_dur.shape)
        # print("\n exceeds_energy:", exceeds_energy.shape)
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | exceeds_dur | exceeds_energy # [batch_size, 1, n_loc]

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a[torch.arange(batch_size), veh] == 0)[:, None] & (
                    (mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]

        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size]

    
    def construct_solutions(self, actions):
        return actions
