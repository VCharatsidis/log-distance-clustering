import sys
import torch


def three_variate_IID_loss(x_1, x_2, x_3, EPS=sys.float_info.epsilon):
  # has had softmax applied
  k = 10
  joint_1_2_3 = compute_three_joint(x_1, x_2, x_3)
  assert (joint_1_2_3.size() == (k, k, k))

  p_z = joint_1_2_3.sum(dim=2).view(k, k, 1).expand(k, k, k)
  p_i = joint_1_2_3.sum(dim=1).view(k, 1, k).expand(k, k, k)
  p_j = joint_1_2_3.sum(dim=0).view(1, k, k).expand(k, k, k)  # but should be same, symmetric

  p_i_j = compute_joint(x_1, x_2)
  p_j_z = compute_joint(x_2, x_3)
  p_i_z = compute_joint(x_1, x_3)

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

  numerator = torch.log(p_i_j) + torch.log(p_i_z) + torch.log(p_j_z)
  denominator = - joint_1_2_3 - torch.log(p_i) - torch.log(p_j) - torch.log(p_z)

  loss = - joint_1_2_3 * (numerator - denominator)

  return loss


def joint(x_1, x_2, x_3):
  # produces variable that requires grad (since args require grad)

  # bn, k = x_out.size()
  # assert (x_2.size(0) == bn and x_2.size(1) == k)
  # assert (x_3.size(1) == k)

  #print("x_1: ", x_1.shape)
  #print("x_2.unsq.t(): ", x_2.unsqueeze(0).t().shape)
  combine_1_2 = x_1 * x_2.unsqueeze(0).t()  # k, k

  x_3_unsq = x_3.unsqueeze(0).unsqueeze(0).transpose(0, 2)
  #print("x_3_unsq: ", x_3_unsq.shape)

  combine_1_2_3 = combine_1_2 * x_3_unsq
  #print("combine_1_2_3: ", combine_1_2_3.shape)

  return combine_1_2_3


def compute_three_joint(x_1, x_2, x_3):
  joint_1_2_3 = joint(x_1, x_2, x_3)

  # p_i_j = p_i_j.sum(dim=0)  # k, k
  # p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  # p_i_j = p_i_j / p_i_j.sum()  # normalise

  return joint_1_2_3


def compute_joint(x_out, x_tf_out):
  p_i_j = x_out * x_tf_out.unsqueeze(0).t()  # k, k

  return p_i_j