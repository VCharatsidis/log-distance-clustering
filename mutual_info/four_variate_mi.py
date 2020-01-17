import sys
import torch
import random


def four_variate_IID_loss(x_1, x_2, x_3, x_4, EPS=sys.float_info.epsilon):
  # has had softmax applied
  k = 10
  joint_probability_1_2_3_4 = joint_probability(x_1, x_2, x_3, x_4)
  assert (joint_probability_1_2_3_4.size() == (k, k, k, k))

  # sum_i = joint_1_2_3.sum(dim=1).sum(dim=1).view(k, 1, 1)
  # sum_j = joint_1_2_3.sum(dim=0).sum(dim=1).view(1, k, 1)
  # sum_z = joint_1_2_3.sum(dim=0).sum(dim=0).view(1, 1, k)

  # if random.uniform(0, 1) > 0.99:
  #   print(sum_i)
  #   print(sum_j)
  #   print(sum_z)

  p_1 = joint_probability_1_2_3_4.sum(dim=1).sum(dim=1).sum(dim=1).view(k, 1, 1, 1).expand(k, k, k, k)
  p_2 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=1).sum(dim=1).view(1, k, 1, 1).expand(k, k, k, k)
  p_3 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=1).view(1, 1, k, 1).expand(k, k, k, k)
  p_4 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=0).view(1, 1, 1, k).expand(k, k, k, k)

  marginal_4 = joint_probability_1_2_3_4.sum(dim=3).view(k, k, k, 1).expand(k, k, k, k)
  marginal_3 = joint_probability_1_2_3_4.sum(dim=2).view(k, k, 1, k).expand(k, k, k, k)
  marginal_2 = joint_probability_1_2_3_4.sum(dim=1).view(k, 1, k, k).expand(k, k, k, k)
  marginal_1 = joint_probability_1_2_3_4.sum(dim=0).view(1, k, k, k).expand(k, k, k, k)

  p_1_2 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=2).view(k, k, 1, 1).expand(k, k, k, k)
  p_1_3 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=1).view(k, 1, k, 1).expand(k, k, k, k)
  p_1_4 = joint_probability_1_2_3_4.sum(dim=2).sum(dim=1).view(k, 1, 1, k).expand(k, k, k, k)

  p_2_3 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=0).view(1, k, k, 1).expand(k, k, k, k)
  p_2_4 = joint_probability_1_2_3_4.sum(dim=2).sum(dim=0).view(1, k, 1, k).expand(k, k, k, k)

  p_3_4 = joint_probability_1_2_3_4.sum(dim=1).sum(dim=0).view(1, 1, k, k).expand(k, k, k, k)

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway

  joint_probability_1_2_3_4[joint_probability_1_2_3_4 < EPS] = EPS
  p_1[(p_1 < EPS).data] = EPS
  p_2[(p_2 < EPS).data] = EPS
  p_3[(p_3 < EPS).data] = EPS
  p_4[(p_4 < EPS).data] = EPS

  # marginal_4[(marginal_4 < EPS).data] = EPS
  # marginal_3[(marginal_3 < EPS).data] = EPS
  # marginal_2[(marginal_2 < EPS).data] = EPS
  # marginal_1[(marginal_1 < EPS).data] = EPS
  #
  # p_1_2[(p_1_2 < EPS).data] = EPS
  # p_1_3[(p_1_3 < EPS).data] = EPS
  # p_1_4[(p_1_4 < EPS).data] = EPS
  #
  # p_2_3[(p_2_3 < EPS).data] = EPS
  # p_2_4[(p_2_4 < EPS).data] = EPS
  #
  # p_3_4[(p_3_4 < EPS).data] = EPS

  # mvmi = multi_variate_mutual_info(joint_probability_1_2_3_4,
  #                               p_1_2,
  #                               p_1_3,
  #                               p_1_4,
  #                               p_2_3,
  #                               p_2_4,
  #                               p_3_4,
  #                               marginal_4,
  #                               marginal_3,
  #                               marginal_2,
  #                               marginal_1,
  #                               p_1,
  #                               p_2,
  #                               p_3,
  #                               p_4)

  loss = total_correlation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4)
  loss = loss.sum()

  return loss


def dual_total_correlation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4, marginal_1, marginal_2, marginal_3, marginal_4):
    joint_entr = joint_entropy(joint_probability_1_2_3_4)

    conditional_entropy_1 = -joint_probability_1_2_3_4 * torch.log(p_1)
    conditional_entropy_2 = -joint_probability_1_2_3_4 * torch.log(p_2)
    conditional_entropy_3 = -joint_probability_1_2_3_4 * torch.log(p_3)
    conditional_entropy_4 = -joint_probability_1_2_3_4 * torch.log(p_4)

    # conditional_entropy_1 = joint_entr - (- p_1 * torch.log(p_1))
    # conditional_entropy_2 = joint_entr - (- p_2 * torch.log(p_2))
    # conditional_entropy_3 = joint_entr - (- p_3 * torch.log(p_3))
    # conditional_entropy_4 = joint_entr - (- p_4 * torch.log(p_4))

    conditional_entropies = conditional_entropy_1 + conditional_entropy_2 + conditional_entropy_3 + conditional_entropy_4

    dtc = joint_entr - conditional_entropies
    normalised_dtc = dtc / joint_entr

    return normalised_dtc


def reverse_total_crrelation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4):
  numerator = torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)
  denominator = torch.log(joint_probability_1_2_3_4)

  rev = - p_1 * p_2 * p_3 * p_4 * (numerator - denominator)
  return rev

def total_correlation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4):
    numerator = torch.log(joint_probability_1_2_3_4)
    denominator = torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)

    total_corr = - joint_probability_1_2_3_4 * (numerator - denominator)
    return total_corr


def multi_variate_mutual_info(joint_probability_1_2_3_4,
                              p_1_2,
                              p_1_3,
                              p_1_4,
                              p_2_3,
                              p_2_4,
                              p_3_4,
                              p_1_2_3,
                              p_1_2_4,
                              p_1_3_4,
                              p_2_3_4,
                              p_1,
                              p_2,
                              p_3,
                              p_4):
    '''


    '''
    numerator = torch.log(joint_probability_1_2_3_4) + \
                torch.log(p_1_2) + \
                torch.log(p_1_3) + \
                torch.log(p_1_4) + \
                torch.log(p_2_3) + \
                torch.log(p_2_4) + \
                torch.log(p_3_4)

    denominator = torch.log(p_1_2_3) + \
                  torch.log(p_1_2_4) + \
                  torch.log(p_1_3_4) + \
                  torch.log(p_2_3_4) + \
                  torch.log(p_1) + \
                  torch.log(p_2) + \
                  torch.log(p_3) + \
                  torch.log(p_4)

    multi_variate_mi = - joint_probability_1_2_3_4 * (numerator - denominator)

    return multi_variate_mi


def joint_entropy(joint_probability):
  '''
  joint entropy is a measure of the uncertainty associated with a set of variables.
  :return:
  '''

  return - joint_probability * torch.log(joint_probability)


def Information_Quality_Ratio(multi_variate_mi, joint_entropy):
  '''
  This normalized version also known as Information Quality Ratio (IQR)
   which quantifies the amount of information of a variable based on another variable against total uncertainty
  :param multi_variate_mi:
  :param joint_entropy:
  :return:
  '''
  return multi_variate_mi / joint_entropy


def joint_probability(x_1, x_2, x_3, x_4):
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

