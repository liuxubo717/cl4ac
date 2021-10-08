
def get_total_grad_norm(parameters, norm_type=2):
    total_norm = 0
    parameters = list(filter(lambda param: param.grad is not None, parameters))
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1. / norm_type)
