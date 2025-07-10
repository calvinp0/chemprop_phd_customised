import importlib.util

import pytest
from arc.family.family import ReactionFamily, get_reaction_family_products
from arc.reaction import ARCReaction
from arc.species import ARCSpecies

from preprocessing import order_reactions


class TestOrderReactionsReactionfamily:

    def setup_class(self):
        """
        Setup the test class
        """
        RxnFamily = ReactionFamily(label="H_Abstraction")
        self.r1_spc = ARCSpecies(
            label="r1",
            smiles="CC[O]",
            xyz="""
C      -0.78351300    0.11230100   -0.07161500
C       0.71678200   -0.11700100   -0.13851200
O       1.36876800    0.52713900   -1.13995000
H      -1.22173700   -0.44233100    0.76058000
H      -1.00016800    1.17205900    0.06703900
H      -1.25987900   -0.21718700   -0.99560800
H       1.21444900    0.16655600    0.80770000
H       0.96050100   -1.19188200   -0.23137500
                            """,
        )
        self.r2_spc = ARCSpecies(
            label="r2",
            smiles="CCOCC",
            xyz="""
C       2.41451000    0.08167800    0.40646700
C       1.12573900   -0.22196700   -0.32186300
O       1.09833100   -1.58793700   -0.65117200
C      -0.06854300   -1.96945000   -1.33521700
C       0.00235700   -3.44780100   -1.64025600
H       3.27356100   -0.15338600   -0.22285700
H       2.45814100    1.13890400    0.67279300
H       2.48550800   -0.50954500    1.32007800
H       0.25972600    0.02944600    0.30828300
H       1.04824900    0.38581800   -1.23557200
H      -0.95348400   -1.74573200   -0.72113100
H      -0.16496100   -1.38936000   -2.26498700
H       0.08313600   -4.02471300   -0.71834600
H      -0.89500200   -3.76743800   -2.17236400
H       0.87118900   -3.66855400   -2.26128100
                            """,
        )
        self.p1_spc = ARCSpecies(
            label="p1",
            smiles="CCO",
            xyz="""
C      -0.98953000    0.30479200    0.10414300
C       0.37607100   -0.35623000    0.07211200
O       0.34410900   -1.65641100   -0.48337000
H      -1.69967300   -0.31137000    0.65723800
H      -0.93202600    1.28483900    0.58221800
H      -1.37736200    0.45065100   -0.90768800
H       0.75808000   -0.48129400    1.08623300
H       1.08997400    0.27868800   -0.46811800
H       0.00892500   -1.59591200   -1.37922000
""",
        )

        self.p2_spc = ARCSpecies(
            label="p2",
            smiles="[CH2]COCC",
            xyz="""
C       2.66940700    0.28135000   -0.60463300
C       1.56260000   -0.50747100   -0.01029800
O       0.55217500   -0.84870300   -0.92778800
C       0.95052000   -1.82039200   -1.86821000
C      -0.21005400   -2.10294400   -2.79296400
H       2.46662300    0.94429600   -1.43548700
H       3.63385200    0.33116700   -0.11789200
H       1.05733100    0.07411800    0.77326200
H       1.95797400   -1.41057800    0.48089600
H       1.81954500   -1.46486500   -2.43786900
H       1.25865700   -2.73715100   -1.34400800
H      -0.50506200   -1.19706700   -3.32375100
H       0.06859300   -2.86009200   -3.52761500
H      -1.06989200   -2.46611700   -2.22887000
""",
        )

        # Reaction 2
        self.r3_spc = ARCSpecies(label='CS', smiles='CS', xyz="""C      -0.50715100   -0.02912500   -0.00965100
S       1.13205100   -0.15269300   -0.76870700
H      -0.61737100   -0.73296600    0.81267000
H      -0.58632300    0.98272500    0.38447000
H      -1.29623200   -0.17999800   -0.74336800
H       1.03124600   -1.42370700   -1.17640700
                                 """)

        self.r4_spc = ARCSpecies(label='C=C[C](C)C', smiles='C=C[C](C)C', xyz="""C       1.07668300   -1.68376800    1.29309500
C       1.00200600   -0.36648300    0.91631200
C       1.39651500    0.72604200    1.68058900
C       1.26661200    2.11974000    1.16188600
C       1.96703800    0.54790300    3.05360400
H       0.74480000   -2.47051200    0.62998700
H       1.46633400   -1.98264200    2.25781400
H       0.59804100   -0.15107100   -0.06946200
H       0.62956000    2.72405200    1.81724500
H       2.24102300    2.62001800    1.13414500
H       0.84262300    2.14481900    0.15785700
H       2.87453300   -0.06344300    3.03282700
H       1.25987300    0.04078200    3.71728200
H       2.22047200    1.50706000    3.50541100
                                 """)

        self.p3_spc = ARCSpecies(label='C=CC(C)C', smiles='C=CC(C)C', xyz="""C       2.12560900    0.48489700   -0.62839000
C       1.11280600    0.21411800    0.17941500
C      -0.30170200   -0.04245600   -0.25133800
C      -1.24274800    1.00746300    0.34232500
C      -0.73809000   -1.45259000    0.15043500
H       3.12434300    0.66014800   -0.24852700
H       1.99085800    0.54111800   -1.70353600
H       1.28879800    0.16645700    1.25328500
H      -0.34259300    0.03426000   -1.34238800
H      -0.95709500    2.01332400    0.03136500
H      -1.21886200    0.97287100    1.43496600
H      -2.27233300    0.82860600    0.02547000
H      -0.69630100   -1.57445100    1.23626900
H      -0.09114500   -2.20791000   -0.29790100
H      -1.76468500   -1.64602200   -0.16755700
                                 """)

        self.p4_spc = ARCSpecies(label='[CH2]S', smiles='[CH2]S', xyz="""C      -0.56419000    0.07937900    0.03476300
S       1.06235100    0.54201300    0.28525600
H      -1.32566400    0.76307500    0.37410500
H      -0.82363900   -0.85191900   -0.44011300
H       1.65114300   -0.53254800   -0.25401000
                                 """)

        # Reaction 3
        self.r5_spc = ARCSpecies(label='CC(C)CO', smiles='CC(C)CO', xyz="""C      -0.69855300    1.32486000   -0.16411500
C      -0.25895500   -0.03436500    0.36710900
C      -0.47342300   -0.14288900    1.87327500
C      -0.98414700   -1.17219200   -0.34619500
O      -0.78054300   -1.19113200   -1.74442300
H      -0.12459600    2.13058300    0.29764900
H      -1.75570200    1.49643900    0.05805600
H      -0.57086400    1.38519200   -1.24490400
H       0.81407500   -0.15414700    0.16673800
H      -1.53462900   -0.04659300    2.11924300
H       0.06107600    0.64804900    2.40187000
H      -0.12493100   -1.10225300    2.26181600
H      -0.69179200   -2.13157700    0.10107700
H      -2.06288900   -1.06273600   -0.20555100
H       0.15284800   -1.32982000   -1.91250500
                                    """)

        self.r6_spc = ARCSpecies(label='O', smiles='[O]', xyz="""O       0.00000000    0.00000000    0.00000000
                                    """)

        self.p5_spc = ARCSpecies(label='OH', smiles='[OH]', xyz="""O       0.00000000    0.00000000    0.48548100
H       0.00000000    0.00000000   -0.48548100
                                    """)

        self.p6_spc = ARCSpecies(label='C[C](C)CO', smiles='C[C](C)CO', xyz="""C      -0.02632400    1.38238400   -0.70058200
C      -0.20394700   -0.04533700   -0.32127600
C      -1.11213300   -0.37822800    0.81043900
C       0.83130200   -1.03811000   -0.73217400
O       0.33072900   -2.35413300   -0.90052700
H       0.60711000    1.91814000    0.02309500
H      -0.98305300    1.91089200   -0.72112900
H       0.44746800    1.48988200   -1.67875000
H      -2.08642900    0.10513700    0.69841000
H      -1.26228600   -1.45407700    0.90034600
H      -0.69677500   -0.02347400    1.76624200
H       1.60884500   -1.13379500    0.03912600
H       1.33502200   -0.69252500   -1.64469500
H      -0.39971300   -2.31024500   -1.52083200
                                 """)

        # Reaction 4

        self.r7_spc = ARCSpecies(label='C=CC=C', smiles='C=CC=C', xyz="""C       1.83343700    0.08472000   -0.10618700
C       0.59932700   -0.40714800   -0.07503300
C      -0.59932700    0.40714800    0.07503300
C      -1.83343700   -0.08472000    0.10618700
H       2.01785900    1.14982400   -0.01889400
H       2.69717100   -0.55720600   -0.21943600
H       0.44653400   -1.47925600   -0.16494300
H      -0.44653400    1.47925600    0.16494300
H      -2.01785900   -1.14982400    0.01889400
H      -2.69717100    0.55720600    0.21943700
                                 """)

        self.r8_spc = ARCSpecies(label='C[C](C)c1ccccc1', smiles='C[C](C)c1ccccc1', xyz="""C      -2.47742300    0.80832700    0.14876500
C      -1.41721400   -0.16963200   -0.24758000
C      -0.29036800    0.29320200   -1.11564500
C      -1.47968600   -1.52037600    0.19466000
C      -0.49070700   -2.46639000   -0.16500400
C      -0.55588200   -3.77497500    0.26645000
C      -1.60380800   -4.20408100    1.07331900
C      -2.58988400   -3.29590500    1.44235500
C      -2.53391500   -1.98515200    1.01622500
H      -3.46968400    0.49334100   -0.19238200
H      -2.54335700    0.91837700    1.23675500
H      -2.28291300    1.79441800   -0.27058200
H      -0.24521200   -0.26615000   -2.05653100
H      -0.39045500    1.34866800   -1.36471300
H       0.68111400    0.15893500   -0.62740900
H       0.33730200   -2.16349600   -0.79178900
H       0.21865300   -4.47253400   -0.02811300
H      -1.65133100   -5.23161700    1.40974000
H      -3.41155500   -3.61750900    2.07059800
H      -3.31636300   -1.30294600    1.32048200
                                 """)

        self.p7_spc = ARCSpecies(label='CC(C)c1ccccc1', smiles='CC(C)c1ccccc1', xyz="""C      -1.36945600    1.33667700   -1.07186500
C      -1.40906200   -0.09805800   -0.54277100
C      -0.18185800   -0.39994200    0.31909500
C      -1.56364000   -1.10277000   -1.66517800
C      -0.59202100   -1.23137300   -2.65530600
C      -0.73532000   -2.15086000   -3.68094100
C      -1.85822200   -2.96421100   -3.73768300
C      -2.83247300   -2.84753300   -2.76019700
C      -2.68278200   -1.92413100   -1.73513200
H      -0.49204600    1.49744400   -1.70253400
H      -2.25529300    1.56071400   -1.66794300
H      -1.32109400    2.04958900   -0.24613700
H      -2.29297600   -0.19183500    0.09483100
H      -0.12152300    0.29546100    1.15884700
H      -0.22000200   -1.41548400    0.71586800
H       0.73827500   -0.30164900   -0.26153500
H       0.29149800   -0.60398100   -2.62635000
H       0.03262700   -2.23376000   -4.44011300
H      -1.97115300   -3.68330300   -4.53905200
H      -3.71348700   -3.47648500   -2.79323900
H      -3.45085600   -1.84001600   -0.97433500
                                 """)

        self.p8_spc = ARCSpecies(label='C=[C]C=C', smiles='C=[C]C=C', xyz="""C       1.89633600   -0.14049000   -0.21499900
C       0.63396100    0.14855900   -0.22543700
C      -0.67695800    0.45005500   -0.23732400
C      -1.65999500   -0.23031800    0.46624200
H       2.32735900   -0.83543500   -0.93133600
H       2.57513600    0.29880700    0.51171700
H      -0.97228000    1.29655600   -0.85196700
H      -2.69341700    0.07774400    0.40154700
H      -1.41398900   -1.07821100    1.09044700
                                 """)

        # Reaction 5
        self.r9_spc = ARCSpecies(label='[N-]=[NH2+]', smiles='[N-]=[NH2+]', xyz="""N       0.00000000   -0.00000000    0.88374200
N       0.00000000    0.00000000   -0.56969800
H       0.00000000    0.87199100   -1.09915400
H      -0.00000000   -0.87199100   -1.09915400
""")
        self.r10_spc = ARCSpecies(label='[N]N', smiles='[N]N', xyz="""N      -0.06458500    0.84004500    0.00000000
N      -0.06458500   -0.57995500   -0.00000000
H       0.45209300   -0.91031300    0.81379900
H       0.45209300   -0.91031300   -0.81379900
""")
        self.p9_spc = ARCSpecies(label='[N]=N', smiles='[N]=N', xyz="""N       0.06364100    0.74080200    0.00000000
N       0.06364100   -0.60119800    0.00000000
H      -0.89096800   -0.97722800    0.00000000
""")
        self.p10_spc = ARCSpecies(label='[NH]N', smiles='[NH]N', xyz="""N      -0.77528400   -0.14671600    0.07202700
N       0.66011000    0.05670300   -0.13268500
H      -1.28952900    0.68285700   -0.25268000
H       1.12591400   -0.81050000    0.13647100
H       0.96983500    0.75773400    0.54081300
""")

    def test_get_species_mapping_reactants(self):
        """
        Test get_species_mapping for reactants

        1R-2H + 3R* <=> 1R* + 2H-3R
        
        """
        rxn01 = ARCReaction(
            r_species=[self.r1_spc, self.r2_spc], p_species=[self.p1_spc, self.p2_spc]
        )
        gen_rxn01_products = get_reaction_family_products(rxn01)
        r_label_maps_rxn01 = [r_maps["r_label_map"] for r_maps in gen_rxn01_products]

        expected_r_label_maps_rxn01 = [
            {"*3": 2, "*1": 8, "*2": 13},
            {"*3": 2, "*1": 8, "*2": 14},
            {"*3": 2, "*1": 8, "*2": 15},
            {"*3": 2, "*1": 12, "*2": 20},
            {"*3": 2, "*1": 12, "*2": 21},
            {"*3": 2, "*1": 12, "*2": 22},
        ]

        assert r_label_maps_rxn01 == expected_r_label_maps_rxn01
        for gen_rxn01_product in gen_rxn01_products:
            r_label_map = gen_rxn01_product["r_label_map"]
            if list(r_label_map.keys())[0] == "*3":
                assert (str(self.r2_spc.mol.atoms[r_label_map["*2"] - len(self.r1_spc.mol.atoms)]) == "H")
                expected_species_mapping = {"Y_rad": self.r1_spc, "X_H": self.r2_spc}
            species_mapped = order_reactions.get_species_mapping(
                r_label_map, self.r1_spc, self.r2_spc
            )

            expected_species_smiles = [
                spec.mol.to_smiles() for spec in expected_species_mapping.values()
            ]
            spec_mapped_smiles = [
                spec.mol.to_smiles() for spec in species_mapped.values()
            ]

            # Assert that the order of the list of species is the same
            assert expected_species_smiles == spec_mapped_smiles

        rxn02 = ARCReaction(
            r_species=[self.r3_spc, self.r4_spc], p_species=[self.p3_spc, self.p4_spc]
        )
        gen_rxn02_products = get_reaction_family_products(rxn02)
        r_label_maps_rxn02 = [r_maps["r_label_map"] for r_maps in gen_rxn02_products]

        expected_r_label_maps_rxn02 = [{'*1': 0, '*2': 2, '*3': 8}, {'*1': 0, '*2': 3, '*3': 8},
                                       {'*1': 0, '*2': 4, '*3': 8}]

        assert r_label_maps_rxn02 == expected_r_label_maps_rxn02

        for gen_rxn02_product in gen_rxn02_products:
            r_label_map = gen_rxn02_product["r_label_map"]
            if list(r_label_map.keys())[0] == "*1":
                assert (str(self.r3_spc.mol.atoms[r_label_map["*2"]]) == "H")
                expected_species_mapping = {"Y_rad": self.r3_spc, "X_H": self.r4_spc}
            species_mapped = order_reactions.get_species_mapping(
                r_label_map, self.r3_spc, self.r4_spc
            )

            expected_species_smiles = [
                spec.mol.to_smiles() for spec in expected_species_mapping.values()
            ]
            spec_mapped_smiles = [
                spec.mol.to_smiles() for spec in species_mapped.values()
            ]

            # Assert that the order of the list of species is the same
            assert expected_species_smiles == spec_mapped_smiles

        rxn03 = ARCReaction(
            r_species=[self.r5_spc, self.r6_spc], p_species=[self.p5_spc, self.p6_spc]
        )
        gen_rxn03_products = get_reaction_family_products(rxn03)
        r_label_maps_rxn03 = [r_maps["r_label_map"] for r_maps in gen_rxn03_products]

        expected_r_label_maps_rxn03 = [{'*1': 1, '*2': 8, '*3': 15}]

        assert r_label_maps_rxn03 == expected_r_label_maps_rxn03

        for gen_rxn03_product in gen_rxn03_products:
            r_label_map = gen_rxn03_product["r_label_map"]
            if list(r_label_map.keys())[0] == "*1":
                assert (str(self.r5_spc.mol.atoms[r_label_map["*2"]]) == "H")
                expected_species_mapping = {"Y_rad": self.r5_spc, "X_H": self.r6_spc}
            species_mapped = order_reactions.get_species_mapping(
                r_label_map, self.r5_spc, self.r6_spc
            )

            expected_species_smiles = [
                spec.mol.to_smiles() for spec in expected_species_mapping.values()
            ]
            spec_mapped_smiles = [
                spec.mol.to_smiles() for spec in species_mapped.values()
            ]

            # Assert that the order of the list of species is the same
            assert expected_species_smiles == spec_mapped_smiles

        rxn04 = ARCReaction(
            r_species=[self.r7_spc, self.r8_spc], p_species=[self.p7_spc, self.p8_spc]
        )
        gen_rxn04_products = get_reaction_family_products(rxn04)
        r_label_maps_rxn04 = [r_maps["r_label_map"] for r_maps in gen_rxn04_products]

        expected_r_label_maps_rxn04 = [{'*1': 1, '*2': 6, '*3': 11}, {'*1': 2, '*2': 7, '*3': 11}]

        assert r_label_maps_rxn04 == expected_r_label_maps_rxn04

        for gen_rxn04_product in gen_rxn04_products:
            r_label_map = gen_rxn04_product["r_label_map"]
            if list(r_label_map.keys())[0] == "*1":
                assert (str(self.r7_spc.mol.atoms[r_label_map["*2"]]) == "H")
                expected_species_mapping = {"Y_rad": self.r7_spc, "X_H": self.r8_spc}
            species_mapped = order_reactions.get_species_mapping(
                r_label_map, self.r7_spc, self.r8_spc
            )

            expected_species_smiles = [
                spec.mol.to_smiles() for spec in expected_species_mapping.values()
            ]
            spec_mapped_smiles = [
                spec.mol.to_smiles() for spec in species_mapped.values()
            ]

            # Assert that the order of the list of species is the same
            assert expected_species_smiles == spec_mapped_smiles

        rxn05 = ARCReaction(
            r_species=[self.r9_spc, self.r10_spc], p_species=[self.p9_spc, self.p10_spc]
        )

        gen_rxn05_products = get_reaction_family_products(rxn05)

        r_label_maps_rxn05 = [r_maps["r_label_map"] for r_maps in gen_rxn05_products]

        expected_r_label_maps_rxn05 = [{'*1': 1, '*2': 2, '*3': 4},
                                       {'*1': 1, '*2': 3, '*3': 4},
                                       {'*1': 1, '*2': 3, '*3': 4, '*4': 5}]

        assert r_label_maps_rxn05 == expected_r_label_maps_rxn05

        for gen_rxn05_product in gen_rxn05_products:
            r_label_map = gen_rxn05_product["r_label_map"]
            if list(r_label_map.keys())[0] == "*1":
                assert (str(self.r9_spc.mol.atoms[r_label_map["*2"]]) == "H")
                expected_species_mapping = {"Y_rad": self.r9_spc, "X_H": self.r10_spc}
            species_mapped = order_reactions.get_species_mapping(
                r_label_map, self.r9_spc, self.r10_spc
            )

            expected_species_smiles = [
                spec.mol.to_smiles() for spec in expected_species_mapping.values()
            ]
            spec_mapped_smiles = [
                spec.mol.to_smiles() for spec in species_mapped.values()
            ]

            # Assert that the order of the list of species is the same
            assert expected_species_smiles == spec_mapped_smiles

    def test_get_species_mapping_products(self):
        """
        Test get_species_mapping for products

        1R-2H + 3R* <=> 1R* + 2H-3R

        """
        rxn01 = ARCReaction(
            r_species=[self.r1_spc, self.r2_spc], p_species=[self.p1_spc, self.p2_spc]
        )
        gen_rxn01_products = get_reaction_family_products(rxn01)
        p_label_maps_rxn01 = [p_maps["p_label_map"] for p_maps in gen_rxn01_products]

        expected_p_label_maps_rxn01_opt_1 = [
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 1, "*3": 16, "*2": 22},
            {"*1": 1, "*3": 16, "*2": 22},
            {"*2": 0, "*3": 1, "*1": 13},
        ]

        expected_p_label_maps_rxn01_opt_2 = [
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 9, "*3": 16, "*2": 22},
            {"*1": 1, "*3": 16, "*2": 22},
            {"*1": 1, "*3": 16, "*2": 22},
            {"*2": 0, "*3": 1, "*1": 13},
        ]

        for orig_p_label_map, opt_1_exp, opt_2_exp in zip(p_label_maps_rxn01, expected_p_label_maps_rxn01_opt_1,
                                                          expected_p_label_maps_rxn01_opt_2):
            assert orig_p_label_map in [opt_1_exp,
                                        opt_2_exp], f"orig_p_label_map: {orig_p_label_map}, opt_1_exp: {opt_1_exp}, opt_2_exp: {opt_2_exp} - An order change may have occured"

        for gen_rxn01_product in gen_rxn01_products:
            p_label_map = gen_rxn01_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn01_product['products']
            if list(p_label_map.keys())[0] == "*1":
                assert (str(p2_gen.atoms[p_label_map["*2"] - len(p1_gen.atoms)]) == "H")
                assert str((p1_gen.atoms[p_label_map["*1"]])).endswith(".")
                expected_species_mapping = {"Y_rad": p1_gen, "X_H": p2_gen}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                assert str(p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith(".")
                expected_species_mapping = {"Y_rad": p2_gen, "X_H": p1_gen}
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            expected_species_smiles = {key: spec.to_smiles() for key, spec in expected_species_mapping.items()}
            spec_mapped_smiles = {key: spec.to_smiles() for key, spec in species_mapped.items()}

            # Assert that the order of the list of species is the same
            for key in expected_species_smiles:
                assert key in spec_mapped_smiles, f"Key '{key}' not found in spec_mapped_smiles - {p_label_map}"
                assert expected_species_smiles[key] == spec_mapped_smiles[key], \
                    f"For key '{key}', expected: {expected_species_smiles[key]}, got: {spec_mapped_smiles[key]} - {p_label_map}"

        rxn02 = ARCReaction(
            r_species=[self.r3_spc, self.r4_spc], p_species=[self.p3_spc, self.p4_spc]
        )
        gen_rxn02_products = get_reaction_family_products(rxn02)
        p_label_maps_rxn02 = [p_maps["p_label_map"] for p_maps in gen_rxn02_products]

        expected_p_label_maps_rxn02 = [{'*3': 2, '*2': 7, '*1': 15}, {'*3': 2, '*2': 7, '*1': 15},
                                       {'*3': 2, '*2': 7, '*1': 15}]

        assert p_label_maps_rxn02 == expected_p_label_maps_rxn02

        for gen_rxn02_product in gen_rxn02_products:
            p_label_map = gen_rxn02_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn02_product['products']
            if list(p_label_map.keys())[0] == "*1":
                assert (str(p2_gen.atoms[p_label_map["*2"] - len(p1_gen.atoms)]) == "H")
                assert str(p1_gen.atoms[p_label_map["*1"]]).endswith(".")
                expected_species_mapping = {"Y_rad": p1_gen, "X_H": p2_gen}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                assert str(p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith(".")
                expected_species_mapping = {"Y_rad": p2_gen, "X_H": p1_gen}
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            expected_species_smiles = {key: spec.to_smiles() for key, spec in expected_species_mapping.items()}
            spec_mapped_smiles = {key: spec.to_smiles() for key, spec in species_mapped.items()}

            # Assert that the order of the list of species is the same
            for key in expected_species_smiles:
                assert key in spec_mapped_smiles, f"Key '{key}' not found in spec_mapped_smiles - {p_label_map}"
                assert expected_species_smiles[key] == spec_mapped_smiles[key], \
                    f"For key '{key}', expected: {expected_species_smiles[key]}, got: {spec_mapped_smiles[key]} - {p_label_map}"

        rxn03 = ARCReaction(
            r_species=[self.r5_spc, self.r6_spc], p_species=[self.p5_spc, self.p6_spc]
        )
        gen_rxn03_products = get_reaction_family_products(rxn03)
        p_label_maps_rxn03 = [p_maps["p_label_map"] for p_maps in gen_rxn03_products]

        expected_p_label_maps_rxn03 = [{'*3': 0, '*2': 1, '*1': 3}]
        assert p_label_maps_rxn03 == expected_p_label_maps_rxn03

        for gen_rxn03_product in gen_rxn03_products:
            p_label_map = gen_rxn03_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn03_product['products']
            if list(p_label_map.keys())[0] == "*1":
                assert (str(p2_gen.atoms[p_label_map["*2"] - len(p1_gen.atoms)]) == "H")
                assert str(p1_gen.atoms[p_label_map["*1"]]).endswith(".")
                expected_species_mapping = {"Y_rad": p1_gen, "X_H": p2_gen}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                assert str(p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith(".")
                expected_species_mapping = {"Y_rad": p2_gen, "X_H": p1_gen}
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            expected_species_smiles = {key: spec.to_smiles() for key, spec in expected_species_mapping.items()}
            spec_mapped_smiles = {key: spec.to_smiles() for key, spec in species_mapped.items()}

            # Assert that the order of the list of species is the same
            for key in expected_species_smiles:
                assert key in spec_mapped_smiles, f"Key '{key}' not found in spec_mapped_smiles - {p_label_map}"
                assert expected_species_smiles[key] == spec_mapped_smiles[key], \
                    f"For key '{key}', expected: {expected_species_smiles[key]}, got: {spec_mapped_smiles[key]} - {p_label_map}"

        rxn04 = ARCReaction(
            r_species=[self.r7_spc, self.r8_spc], p_species=[self.p7_spc, self.p8_spc]
        )

        gen_rxn04_products = get_reaction_family_products(rxn04)

        p_label_maps_rxn04 = [p_maps["p_label_map"] for p_maps in gen_rxn04_products]

        expected_p_label_maps_rxn04 = [{'*3': 4, '*2': 10, '*1': 22}, {'*3': 4, '*2': 10, '*1': 23}]

        assert p_label_maps_rxn04 == expected_p_label_maps_rxn04

        for gen_rxn04_product in gen_rxn04_products:
            p_label_map = gen_rxn04_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn04_product['products']
            if list(p_label_map.keys())[0] == "*1":
                assert (str(p2_gen.atoms[p_label_map["*2"] - len(p1_gen.atoms)]) == "H")
                assert str(p1_gen.atoms[p_label_map["*1"]]).endswith(".")
                expected_species_mapping = {"Y_rad": p1_gen, "X_H": p2_gen}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                assert str(p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith(".")
                expected_species_mapping = {"Y_rad": p2_gen, "X_H": p1_gen}
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            expected_species_smiles = {key: spec.to_smiles() for key, spec in expected_species_mapping.items()}
            spec_mapped_smiles = {key: spec.to_smiles() for key, spec in species_mapped.items()}

            # Assert that the order of the list of species is the same
            for key in expected_species_smiles:
                assert key in spec_mapped_smiles, f"Key '{key}' not found in spec_mapped_smiles - {p_label_map}"
                assert expected_species_smiles[key] == spec_mapped_smiles[key], \
                    f"For key '{key}', expected: {expected_species_smiles[key]}, got: {spec_mapped_smiles[key]} - {p_label_map}"

        rxn05 = ARCReaction(
            r_species=[self.r9_spc, self.r10_spc], p_species=[self.p9_spc, self.p10_spc]
        )

        gen_rxn05_products = get_reaction_family_products(rxn05)

        p_label_maps_rxn05 = [p_maps["p_label_map"] for p_maps in gen_rxn05_products]

        expected_p_label_maps_rxn05 = [{'*3': 1, '*2': 2, '*1': 5, '*4': 7},
                                       {'*3': 2, '*2': 4, '*1': 6},
                                       {'*3': 2, '*2': 4, '*1': 6}]

        assert p_label_maps_rxn05 == expected_p_label_maps_rxn05

        for gen_rxn05_product in gen_rxn05_products:
            p_label_map = gen_rxn05_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn05_product['products']
            if list(p_label_map.keys())[0] == "*1":
                assert (str(p2_gen.atoms[p_label_map["*2"] - len(p1_gen.atoms)]) == "H")
                assert str(p1_gen.atoms[p_label_map["*1"]]).endswith(".")
                expected_species_mapping = {"Y_rad": p1_gen, "X_H": p2_gen}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                assert str(p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith(".") or str(
                    p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith("+") or str(
                    p2_gen.atoms[p_label_map["*1"] - len(p1_gen.atoms)]).endswith("-")
                expected_species_mapping = {"Y_rad": p2_gen, "X_H": p1_gen}
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            expected_species_smiles = {key: spec.to_smiles() for key, spec in expected_species_mapping.items()}
            spec_mapped_smiles = {key: spec.to_smiles() for key, spec in species_mapped.items()}

            # Assert that the order of the list of species is the same
            for key in expected_species_smiles:
                assert key in spec_mapped_smiles, f"Key '{key}' not found in spec_mapped_smiles - {p_label_map}"
                assert expected_species_smiles[key] == spec_mapped_smiles[key], \
                    f"For key '{key}', expected: {expected_species_smiles[key]}, got: {spec_mapped_smiles[key]} - {p_label_map}"

    def test_replace_molecule_arcspecies(self):
        """
        Test the replace_molecule_arcspecies function
        """
        rxn01 = ARCReaction(
            r_species=[self.r1_spc, self.r2_spc], p_species=[self.p1_spc, self.p2_spc]
        )
        gen_rxn01_products = get_reaction_family_products(rxn01)
        user_products = [self.p1_spc, self.p2_spc]

        for gen_rxn01_product in gen_rxn01_products:
            p_label_map = gen_rxn01_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn01_product['products']
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            order_reactions.replace_molecule_arcspecies(
                species_mapped, user_products
            )

            expected_species_mapping = {"Y_rad": self.p2_spc, "X_H": self.p1_spc}

            assert species_mapped['X_H'] == expected_species_mapping['X_H']
            assert species_mapped['Y_rad'] == expected_species_mapping['Y_rad']

        rxn02 = ARCReaction(
            r_species=[self.r3_spc, self.r4_spc], p_species=[self.p3_spc, self.p4_spc]
        )
        gen_rxn02_products = get_reaction_family_products(rxn02)
        user_products = [self.p3_spc, self.p4_spc]

        for gen_rxn02_product in gen_rxn02_products:
            p_label_map = gen_rxn02_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn02_product['products']
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            order_reactions.replace_molecule_arcspecies(
                species_mapped, user_products
            )

            if list(p_label_map.keys())[0] == "*1":
                expected_species_mapping = {"Y_rad": self.p3_spc, "X_H": self.p4_spc}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                expected_species_mapping = {"Y_rad": self.p4_spc, "X_H": self.p3_spc}

            assert species_mapped['X_H'] == expected_species_mapping['X_H']
            assert species_mapped['Y_rad'] == expected_species_mapping['Y_rad']

        rxn03 = ARCReaction(
            r_species=[self.r5_spc, self.r6_spc], p_species=[self.p5_spc, self.p6_spc]
        )
        gen_rxn03_products = get_reaction_family_products(rxn03)
        user_products = [self.p5_spc, self.p6_spc]

        for gen_rxn03_product in gen_rxn03_products:
            p_label_map = gen_rxn03_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn03_product['products']
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            order_reactions.replace_molecule_arcspecies(
                species_mapped, user_products
            )

            expected_species_mapping = {"Y_rad": self.p6_spc, "X_H": self.p5_spc}

            assert species_mapped['X_H'] == expected_species_mapping['X_H']
            assert species_mapped['Y_rad'] == expected_species_mapping['Y_rad']

        rxn04 = ARCReaction(
            r_species=[self.r7_spc, self.r8_spc], p_species=[self.p7_spc, self.p8_spc]
        )
        gen_rxn04_products = get_reaction_family_products(rxn04)
        user_products = [self.p7_spc, self.p8_spc]

        for gen_rxn04_product in gen_rxn04_products:
            p_label_map = gen_rxn04_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn04_product['products']
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            order_reactions.replace_molecule_arcspecies(
                species_mapped, user_products
            )

            if list(p_label_map.keys())[0] == "*1":
                expected_species_mapping = {"Y_rad": self.p7_spc, "X_H": self.p8_spc}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                expected_species_mapping = {"Y_rad": self.p8_spc, "X_H": self.p7_spc}

            assert species_mapped['X_H'] == expected_species_mapping['X_H']
            assert species_mapped['Y_rad'] == expected_species_mapping['Y_rad']

        rxn05 = ARCReaction(
            r_species=[self.r9_spc, self.r10_spc], p_species=[self.p9_spc, self.p10_spc]
        )

        gen_rxn05_products = get_reaction_family_products(rxn05)

        user_products = [self.p9_spc, self.p10_spc]

        for gen_rxn05_product in gen_rxn05_products:
            p_label_map = gen_rxn05_product["p_label_map"]
            p1_gen, p2_gen = gen_rxn05_product['products']
            species_mapped = order_reactions.get_species_mapping(
                p_label_map, p1_gen, p2_gen, reactants=False
            )

            order_reactions.replace_molecule_arcspecies(
                species_mapped, user_products
            )

            if list(p_label_map.keys())[0] == "*1":
                expected_species_mapping = {"Y_rad": self.p9_spc, "X_H": self.p10_spc}
            elif list(p_label_map.keys())[0] == "*2" or list(p_label_map.keys())[0] == "*3":
                expected_species_mapping = {"Y_rad": self.p10_spc, "X_H": self.p9_spc}

            assert species_mapped['X_H'] == expected_species_mapping['X_H']
            assert species_mapped['Y_rad'] == expected_species_mapping['Y_rad']
