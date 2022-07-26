def adv_attack(model, adv_method, adv_params):
    for k in ['eps','alpha','step_size']:
        if k in adv_params.keys():
            adv_params[k] = eval(adv_params[k])

    # if adv_method in ['CW','DeepFool']:
    #     atk = __import__('adv_attacks').__dict__[adv_method](model=model, **adv_params)
    # else:
    #     atk = __import__('torchattacks').__dict__[adv_method](model=model, **adv_params)

    atk = __import__('torchattacks').__dict__[adv_method](model=model, **adv_params)

    return atk