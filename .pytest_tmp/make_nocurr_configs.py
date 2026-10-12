import os

base = 'experiments/training/RNN'
jobs = [
    ('proTEncGRUpervar_LogNCensSWHM_dyconex_2000ep_warmup_90_to_0_absolute',
     'proTEncGRUpervar_LogNCensSWHM_dyconex_2000ep_nocurr_absolute'),
    ('proTEncGRUpervar_LogNCensSWHM_dyconex_2000ep_warmup_90_to_0_delta',
     'proTEncGRUpervar_LogNCensSWHM_dyconex_2000ep_nocurr_delta'),
]
for src_name, dst_name in jobs:
    src_dir = os.path.join(base, src_name)
    src_file = [f for f in os.listdir(src_dir) if f.startswith('config_') and f.endswith('.yaml')][0]
    with open(os.path.join(src_dir, src_file), encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    # locate the ADAPTIVE TARGET OPTS banner block in the training section
    i_anchor = next(i for i, l in enumerate(lines) if 'ADAPTIVE TARGET OPTS' in l)
    start = i_anchor
    while start > 0 and not lines[start-1].strip().startswith('#_'):
        start -= 1
    start -= 1  # include the opening #___ separator line
    end = next(i for i in range(i_anchor+1, len(lines))
               if lines[i].strip().startswith('#_') and i > i_anchor+3)
    end += 1  # include the closing #___ separator line
    removed = lines[start:end]
    assert any('epoch_show_trg' in l for l in removed), 'block check failed'
    assert any('show_trg_ramp_up_frac' in l for l in removed), 'block check failed'
    assert not any('loss_' in l for l in removed), 'block swallowed loss keys!'
    lines = lines[:start] + lines[end:]
    text = '\n'.join(lines)

    text = (text.replace('warmup_90_to_0_absolute', 'nocurr_absolute')
                .replace('warmup_90_to_0_delta', 'nocurr_delta'))
    note = ('  # ARM "nocurr": NO teacher-forcing curriculum - fully autoregressive\n'
            '  # training from epoch 0 (target_show_mode unset -> show_trg_max_idx\n'
            '  # stays None -> forcing 0.0 everywhere; best-checkpoint selection\n'
            '  # active from epoch 0 since no curriculum warmup applies).\n'
            '  # Rationale: runs 14971469/15045626 trained with zero forcing for\n'
            '  # the first 100 epochs and reached val_mae ~0.06 / val_r2 ~0.7\n'
            '  # within 10-20 epochs - matching the best curriculum run\n'
            '  # (15145912) - while heavy forcing (warmup 0.9) collapsed training\n'
            '  # in both the delta and absolute arms. All other settings are\n'
            '  # identical to the parent warmup_90_to_0 arm.\n\n')
    assert 'experiment:' in text
    text = text.replace('experiment:', note + 'experiment:', 1)

    dst_dir = os.path.join(base, dst_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst_file = (src_file.replace('warmup_90_to_0_absolute', 'nocurr_absolute')
                        .replace('warmup_90_to_0_delta', 'nocurr_delta'))
    out = os.path.join(dst_dir, dst_file)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    print('written', out)
