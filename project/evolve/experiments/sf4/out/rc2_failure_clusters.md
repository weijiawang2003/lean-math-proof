# SF4 RC2 failure clusters

- confirmed failures clustered: **27**
- clusters: **10**

| cluster_id | size | ns | shape | families | risk | reco |
|---|---|---|---|---|---|---|
| `Set__iff__iff` | 16 | Set | iff | set_ext_aesop,set_ext_simp,set_iff_constructor_aesop,set_subset_antisymm | high | sequence_probe |
| `Set__ite_if__subset` | 3 | Set | subset | set_ite_simp_aesop,set_ite_ext,set_subset_antisymm | high | sequence_probe |
| `Multiset__iff__iff` | 1 | Multiset | iff | multiset_tofinset_simp_aesop | medium | sequence_probe |
| `Set__ite_if__equality` | 1 | Set | equality | set_ite_simp_aesop,set_ite_ext,set_ext_aesop,set_ext_simp | low | sequence_probe |
| `Set__singleton__arithmetic` | 1 | Set | arithmetic | set_ext_aesop | low | sequence_probe |
| `unknown__subset__subset` | 1 | unknown | subset | generic_aesop_simpall | high | sequence_probe |
| `unknown__iff__iff` | 1 | unknown | iff | generic_aesop_simpall | high | sequence_probe |
| `unknown__compl__arithmetic` | 1 | unknown | arithmetic | generic_aesop_simpall | high | sequence_probe |
| `Set__other__equality` | 1 | Set | equality | set_ext_aesop,set_ext_simp | low | sequence_probe |
| `Set__map_filter__arithmetic` | 1 | Set | arithmetic | set_ext_aesop | low | sequence_probe |

## Cluster detail

### `Set__iff__iff` (size 16, risk high)
- representatives: ['Set.antitoneOn_iff_antitone', 'Set.diff_singleton_subset_iff', 'Set.pair_eq_pair_iff', 'Set.ssubset_singleton_iff', 'Set.subset_insert_iff']
- common features: ['iff', 'inter/union/diff', 'nonempty', 'singleton', 'subset']
- symptoms: ['aesop failed', 'missing bridge lemma likely', 'simp failed']
- candidate families: ['set_ext_aesop', 'set_ext_simp', 'set_iff_constructor_aesop', 'set_subset_antisymm']
- recommendation: **sequence_probe**

### `Set__ite_if__subset` (size 3, risk high)
- representatives: ['Set.ite_eq_of_subset_left', 'Set.ite_eq_of_subset_right', 'Set.subset_ite']
- common features: ['ite/if', 'subset']
- symptoms: ['aesop failed', 'simp failed']
- candidate families: ['set_ite_simp_aesop', 'set_ite_ext', 'set_subset_antisymm']
- recommendation: **sequence_probe**

### `Multiset__iff__iff` (size 1, risk medium)
- representatives: ['Multiset.toFinset_eq_singleton_iff']
- common features: ['iff', 'singleton', 'toFinset']
- symptoms: ['aesop failed', 'simp failed']
- candidate families: ['multiset_tofinset_simp_aesop']
- recommendation: **sequence_probe**

### `Set__ite_if__equality` (size 1, risk low)
- representatives: ['Set.ite_inter_of_inter_eq']
- common features: ['inter/union/diff', 'ite/if']
- symptoms: ['aesop failed', 'simp failed']
- candidate families: ['set_ite_simp_aesop', 'set_ite_ext', 'set_ext_aesop', 'set_ext_simp']
- recommendation: **sequence_probe**

### `Set__singleton__arithmetic` (size 1, risk low)
- representatives: ['Set.powerset_singleton']
- common features: ['powerset', 'singleton']
- symptoms: ['aesop failed', 'simp failed']
- candidate families: ['set_ext_aesop']
- recommendation: **sequence_probe**

### `unknown__subset__subset` (size 1, risk high)
- representatives: ['Eq.subset']
- common features: ['subset']
- symptoms: ['missing bridge lemma likely']
- candidate families: ['generic_aesop_simpall']
- recommendation: **sequence_probe**

### `unknown__iff__iff` (size 1, risk high)
- representatives: ['Function.Injective.nonempty_apply_iff']
- common features: ['iff', 'nonempty']
- symptoms: ['aesop failed']
- candidate families: ['generic_aesop_simpall']
- recommendation: **sequence_probe**

### `unknown__compl__arithmetic` (size 1, risk high)
- representatives: ['Prop.compl_singleton']
- common features: ['compl', 'singleton']
- symptoms: ['missing bridge lemma likely']
- candidate families: ['generic_aesop_simpall']
- recommendation: **sequence_probe**

### `Set__other__equality` (size 1, risk low)
- representatives: ['Set.eq_of_inclusion_surjective']
- common features: ['other']
- symptoms: ['aesop failed', 'simp failed']
- candidate families: ['set_ext_aesop', 'set_ext_simp']
- recommendation: **sequence_probe**

### `Set__map_filter__arithmetic` (size 1, risk low)
- representatives: ['Set.pairwiseDisjoint_filter']
- common features: ['disjoint', 'map/filter', 'singleton']
- symptoms: ['missing bridge lemma likely']
- candidate families: ['set_ext_aesop']
- recommendation: **sequence_probe**
