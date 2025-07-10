import importlib.util
import ast

import pytest

from arc.species import ARCSpecies

from preprocessing import create_sdf

class TestCreateSdfTest:

    def test_rdkit_conf_from_arcspc(self):
        
        r1h_xyz ={
            'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 1),
            'coords': [[-0.5720214553322238, -1.125130270499479, -0.33699224682334433],
                       [-0.012967455332224098, 0.2802847295005211, -0.46858024682334437],
                       [0.47286454466777594, 0.8050317295005209, 0.7339887531766556],
                       [-1.3531564553322242, -1.156773270499479, 0.4238567531766556],
                       [-0.9973714553322242, -1.459941270499479, -1.2862102468233445],
                       [0.21704054466777611, -1.823355270499479, -0.055347246823344354],
                       [0.759873544667776, 0.312307729500521, -1.2451822468233442],
                       [-0.8022974553322242, 0.9745067295005212, -0.7851522468233443],
                       [1.229262103331989, 0.5653263736143569, 0.8359891237951282]]
        }
        r2h_xyz = {
            'symbols': ('C', 'C', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
            'isotopes': (12, 12, 16, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            'coords': [[0.11743395156648417, 1.8384157556123655, 1.5233495574818559],
                       [0.2448239515664843, 0.3832487556123656, 1.203369557481856],
                       [0.0009549515664841746, 0.18781175561236563, -0.16444844251814408],
                       [0.008321951566484298, -1.1662572443876345, -0.5469084425181441],
                       [-0.3425780484335158, -1.2565992443876344, -2.013115442518144],
                       [-0.6461842262166866, 2.06357326044862, 1.4446642565990229],
                       [0.3201809515664842, 2.0758637556123656, 2.565945557481856],
                       [0.6514489515664843, 2.489276755612366, 0.833536557481856],
                       [1.2572189515664844, 0.02967175561236557, 1.462645557481856],
                       [-0.4619640484335158, -0.20472724438763445, 1.807712557481856],
                       [0.9978159515664842, -1.6051952443876345, -0.3544054425181441],
                       [-0.7199490484335158, -1.7275712443876345, 0.056832557481855894],
                       [0.3803099515664843, -0.7047522443876344, -2.614744442518144],
                       [-0.3428550484335158, -2.2977502443876343, -2.3387014425181443],
                       [-1.3332470484335157, -0.8366212443876344, -2.191873442518144]]
        }
        
        index_dict = {'r1h_a': 2, 'r1h_h': 8, 'r2h_a': 0, 'r2h_h': 5}

        r1h_spc = ARCSpecies(label='r1h', xyz=r1h_xyz)
        mol_r1h = create_sdf.rdkit_conf_from_arcspc(r1h_spc, mol_type='reactant', label_map=index_dict)
        assert mol_r1h.GetNumAtoms() == 9
        assert mol_r1h.GetNumConformers() == 1
        assert mol_r1h.GetAtomWithIdx(2).GetSymbol() == 'O'
        assert mol_r1h.GetAtomWithIdx(8).GetSymbol() == 'H'
        assert ast.literal_eval(mol_r1h.GetProp('mol_properties')) == {'2': {'label': 'donator', 'atom_type': 'O2s'}, '8': {'label': 'd_hydrogen', 'atom_type': 'H0'}}
        assert mol_r1h.GetProp('type') == 'r1h'

        r2h_spc = ARCSpecies(label='r2h', xyz=r2h_xyz)
        mol_r2h = create_sdf.rdkit_conf_from_arcspc(r2h_spc, mol_type='product', label_map=index_dict)
        assert mol_r2h.GetNumAtoms() == 15
        assert mol_r2h.GetNumConformers() == 1
        assert mol_r2h.GetAtomWithIdx(0).GetSymbol() == 'C'
        assert mol_r2h.GetAtomWithIdx(5).GetSymbol() == 'H'
        assert ast.literal_eval(mol_r2h.GetProp('mol_properties')) == {'0': {'label': 'acceptor', 'atom_type': 'Cs'}, '5': {'label': 'a_hydrogen', 'atom_type': 'H0'}}
        assert mol_r2h.GetProp('type') == 'r2h'
