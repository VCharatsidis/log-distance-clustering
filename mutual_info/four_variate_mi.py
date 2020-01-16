import sys
import torch
import random


def four_variate_IID_loss(x_1, x_2, x_3, x_4, EPS=sys.float_info.epsilon):
  # has had softmax applied
  k = 10
  joint_1_2_3_4 = joint(x_1, x_2, x_3, x_4)
  assert (joint_1_2_3_4.size() == (k, k, k, k))

  # sum_i = joint_1_2_3.sum(dim=1).sum(dim=1).view(k, 1, 1)
  # sum_j = joint_1_2_3.sum(dim=0).sum(dim=1).view(1, k, 1)
  # sum_z = joint_1_2_3.sum(dim=0).sum(dim=0).view(1, 1, k)

  # if random.uniform(0, 1) > 0.99:
  #   print(sum_i)
  #   print(sum_j)
  #   print(sum_z)

  p_1 = joint_1_2_3_4.sum(dim=1).sum(dim=1).sum(dim=1).view(k, 1, 1, 1).expand(k, k, k, k)
  p_2 = joint_1_2_3_4.sum(dim=0).sum(dim=1).sum(dim=1).view(1, k, 1, 1).expand(k, k, k, k)
  p_3 = joint_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=1).view(1, 1, k, 1).expand(k, k, k, k)
  p_4 = joint_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=0).view(1, 1, 1, k).expand(k, k, k, k)

  # print("p")
  # print(p_i.shape)

  # input()

  # print(joint_1_2_3.sum(dim=2))
  # print(joint_1_2_3.sum(dim=2).view(k, k, 1).shape)
  # input()

  p_1_2_3 = joint_1_2_3_4.sum(dim=3).view(k, k, k, 1).expand(k, k, k, k)
  p_1_2_4 = joint_1_2_3_4.sum(dim=2).view(k, k, 1, k).expand(k, k, k, k)
  p_1_3_4 = joint_1_2_3_4.sum(dim=1).view(k, 1, k, k).expand(k, k, k, k)
  p_2_3_4 = joint_1_2_3_4.sum(dim=0).view(1, k, k, k).expand(k, k, k, k)

  p_1_2 = joint_1_2_3_4.sum(dim=3).sum(dim=2).view(k, k, 1, 1).expand(k, k, k, k)
  p_1_3 = joint_1_2_3_4.sum(dim=3).sum(dim=1).view(k, 1, k, 1).expand(k, k, k, k)
  p_1_4 = joint_1_2_3_4.sum(dim=2).sum(dim=1).view(k, 1, 1, k).expand(k, k, k, k)

  p_2_3 = joint_1_2_3_4.sum(dim=3).sum(dim=0).view(1, k, k, 1).expand(k, k, k, k)
  p_2_4 = joint_1_2_3_4.sum(dim=2).sum(dim=0).view(1, k, 1, k).expand(k, k, k, k)

  p_3_4 = joint_1_2_3_4.sum(dim=1).sum(dim=0).view(1, 1, k, k).expand(k, k, k, k)

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  # print(joint_1_2_3)
  # print(joint_1_2_3.shape)
  # print(joint_1_2_3[0][0].shape)
  # joint_1_2_3[0][joint_1_2_3[0] < EPS] = EPS
  # joint_1_2_3[:, (joint_1_2_3 < EPS).data, :] = EPS
  # joint_1_2_3[(joint_1_2_3 < EPS).data, :, :] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS
  # p_z[(p_z < EPS).data] = EPS

  # numerator = torch.log(joint_1_2_3_4) + \
  #             torch.log(p_1_2) + \
  #             torch.log(p_1_3) + \
  #             torch.log(p_1_4) + \
  #             torch.log(p_2_3) + \
  #             torch.log(p_2_4) + \
  #             torch.log(p_3_4)
  #
  # denominator = torch.log(p_1_2_3) + \
  #               torch.log(p_1_2_4) + \
  #               torch.log(p_1_3_4) + \
  #               torch.log(p_2_3_4) + \
  #               torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)

  # Total correlation
  numerator = torch.log(joint_1_2_3_4)
  denominator = torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)

  coeff = 1.05
  loss = - joint_1_2_3_4 * (numerator - coeff * denominator)
  loss = loss.sum()
  #loss = torch.abs(loss)
  return loss


def joint(x_1, x_2, x_3, x_4):
  # produces variable that requires grad (since args require grad)

  bn, k = x_1.size()
  assert (x_2.size(0) == bn and x_2.size(1) == k)
  assert (x_3.size(1) == k and x_4.size(1) == k)

  # print("x1", x_1.shape)
  # print("x1", x_1.unsqueeze(2).shape)
  # print("")
  # print("x2", x_2.shape)
  # print("x2", x_2.unsqueeze(1).shape)
  # print("")

  combine_1_2 = x_1.unsqueeze(2) * x_2.unsqueeze(1)  # batch, k, k
  x_3_unsq = x_3.unsqueeze(1).unsqueeze(2)

  combine_1_2_3 = combine_1_2.unsqueeze(3) * x_3_unsq
  x_4_unsq = x_4.unsqueeze(1).unsqueeze(2).unsqueeze(3)

  combine_1_2_3_4 = combine_1_2_3.unsqueeze(4) * x_4_unsq
  combine_1_2_3_4 = combine_1_2_3_4.sum(dim=0)  # k, k, k, k
  combine_1_2_3_4 = combine_1_2_3_4 / combine_1_2_3_4.sum()  # normalise

  return combine_1_2_3_4

