import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

dataset = 'baby'
method = 'feat prop 1'
layers = [1, 2, 3, 20]


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        print('ok')
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


values = {
    "baby":
        {
            "zeros":
                {
                    10: np.array([0.07370885809318069, 0.0744375211251047, 0.07252585222958641, 0.07221724200430096,
                                  0.07277445491106636]),
                    20: np.array([0.07441180360633093, 0.07502473780377288, 0.07519618792893147, 0.07406461710288477,
                                  0.07291590126432222]),
                    30: np.array([0.07388888072459722, 0.07362741928373036, 0.07340575876477533, 0.0749861615256122,
                                  0.07463775037841491]),
                    40: np.array([0.07389316697772617, 0.07321593898334974, 0.07311306890825459, 0.07466897879406881,
                                  0.07360170176495658]),
                    50: np.array([0.0746218300096502, 0.07400889581220822, 0.07480613889419567, 0.07423606722804335,
                                  0.07366170930876208]),
                    60: np.array([0.07304448885819116, 0.0744705865063853, 0.07324594275525248, 0.0740003233059503,
                                  0.07301754669566625]),
                    70: np.array([0.07488941466927271, 0.0749861615256122, 0.07422749472178543, 0.0737486590150925,
                                  0.0748147114004536]),
                    80: np.array([0.07367334342439784, 0.07341310662728211, 0.074278929759333, 0.07464754752842397,
                                  0.07360170176495658]),
                    90: np.array([0.07684639538358291, 0.07574482832943898, 0.07367456806814898, 0.07561195448244104,
                                  0.07482634551608938])
                },
            "mean":
                {
                    10: np.array([0.08489077402383646, 0.08355823915822887, 0.08485391224692737, 0.08349180223472992,
                                  0.08442057205558903]),
                    20: np.array([0.08120196334886182, 0.08296532788611792, 0.08258225932076359, 0.08013731930381453,
                                  0.08120937244355617]),
                    30: np.array([0.08023375999921623, 0.0809667092842692, 0.08178109737877252, 0.07934148456214088,
                                  0.08174815446186703]),
                    40: np.array([0.07856946914142675, 0.07880949931664878, 0.0795613081154692, 0.07870589445530296,
                                  0.0814969800285097]),
                    50: np.array([0.07830065983805311, 0.0779869061090129, 0.07853873058327333, 0.0786011874145811,
                                  0.07648500301262363]),
                    60: np.array([0.07644948834384077, 0.07621974517612826, 0.07713100259134618, 0.07747218834041178,
                                  0.07588125364331516]),
                    70: np.array([0.07660379345648351, 0.07585553612454136, 0.07515075364576444, 0.07758105916988749,
                                  0.07466946865156926]),
                    80: np.array([0.07564073361059268, 0.07481899765358258, 0.07402775532597568, 0.07479328013480879,
                                  0.07568910703876242]),
                    90: np.array([0.07537498591659686, 0.074366491787539, 0.07426913260932395, 0.0744246623657178,
                                  0.07627509907367946])
                },
            "random":
                {
                    10: np.array([0.08482819472815359, 0.08622171685256759, 0.08598076819453222, 0.08608069912462464,
                                  0.08655433009537525]),
                    20: np.array([0.0831153467456317, 0.08321699217697572, 0.08276393522124414, 0.08196718199675714,
                                  0.08322991216855016]),
                    30: np.array([0.08130869105177305, 0.0820563360618396, 0.07954416310295334, 0.08214242851754425,
                                  0.08188274281011654]),
                    40: np.array([0.0772106044351698, 0.07914003066507952, 0.07901523946683908, 0.07916146193072435,
                                  0.07925171817518284]),
                    50: np.array([0.0774023836465972, 0.07660807970961248, 0.07814255832978187, 0.07747647459354075,
                                  0.07714214684948148]),
                    60: np.array([0.07383793554455009, 0.07605209144659818, 0.07472212833286797, 0.0732640062505817,
                                  0.07641397367505792]),
                    70: np.array([0.07511646362073272, 0.07592534081835595, 0.0734100450179043, 0.07374437276196355,
                                  0.072558917610867]),
                    80: np.array([0.07299170671251733, 0.07454345280957769, 0.07458619287649222, 0.07077571384484255,
                                  0.07331023655218696]),
                    90: np.array([0.07220438324491407, 0.07208228626292612, 0.07154674955055573, 0.07216580696675336,
                                  0.07199607134284636])
                },
            "feat prop 1":
                {
                    10: np.array([0.08442834854340873, 0.08546696890874445, 0.08724147770413587, 0.08508243077088874,
                                  0.08529802930327568]),
                    20: np.array([0.08337986979587639, 0.08675866190525175, 0.0850012981223762, 0.08523587863290569,
                                  0.0835889777163823]),
                    30: np.array([0.08179303765534605, 0.08339242239432548, 0.08450164347191401, 0.08283943450850148,
                                  0.08462122993421213]),
                    40: np.array([0.08100057068398803, 0.08271121430775787, 0.08294438647797354, 0.08430153668297892,
                                  0.08211971137596073]),
                    50: np.array([0.08292417985607987, 0.079410799398455, 0.0818741703038586, 0.08030234004927965,
                                  0.07943100602034867]),
                    60: np.array([0.07836372899123645, 0.08033711993181183, 0.07917370836823567, 0.07854589474921746,
                                  0.0793379330952626]),
                    70: np.array([0.07865396956025493, 0.07804826076094463, 0.07625183084240796, 0.07752362337795936,
                                  0.07521639455082517]),
                    80: np.array([0.07460774660651218, 0.07472776169412318, 0.07600506512655468, 0.07461203285964113,
                                  0.07389745323085514]),
                    90: np.array([0.07513054702387076, 0.07378601064950208, 0.07451038742829712, 0.07326614937714619,
                                  0.07507800980694715])
                },
            "feat prop 2":
                {
                    10: np.array([0.08608418935931537, 0.0851069236459114, 0.08625778261103845, 0.08381186287908847,
                                  0.08479868081375129]),
                    20: np.array([0.08270815269838003, 0.08578874405435459, 0.0868577355847184, 0.08614456429624623,
                                  0.08304217428150151]),
                    30: np.array([0.0827377278449699, 0.08513386580843632, 0.0847862506796773, 0.08534089183456534,
                                  0.08319372394570418]),
                    40: np.array([0.08179781376597547, 0.08347159561283622, 0.08244718111501365, 0.08247375588441322,
                                  0.08266271841521301]),
                    50: np.array([0.08015832194414645, 0.08128511665956373, 0.08227879259923288, 0.07994535639582447,
                                  0.07936365061403637]),
                    60: np.array([0.08043521389627757, 0.0798840017438927, 0.08054530936950441, 0.07946621452819373,
                                  0.08044305161628482]),
                    70: np.array([0.08086432906667453, 0.07785366486888963, 0.07713571746978803, 0.07854375162265297,
                                  0.07685668239109243]),
                    80: np.array([0.07537572070284754, 0.07579626336698654, 0.0750166551550154, 0.07533163352780675,
                                  0.07661310074899212]),
                    90: np.array([0.0708878912124463, 0.0710152541625641, 0.06995361049470708, 0.07198015097408166,
                                  0.07107403706261847])
                },
            "feat prop 3":
                {
                    10: np.array([0.08630707452202153, 0.08529123253045688, 0.0870550256930259, 0.08406781342307522,
                                  0.08498249984079631]),
                    20: np.array([0.08346988111158464, 0.08660741840198685, 0.08566781048393023, 0.08615313680250415,
                                  0.08360538794264749]),
                    30: np.array([0.08274630035122782, 0.08431304833423957, 0.08400872436208306, 0.08498170382235808,
                                  0.08350711028161908]),
                    40: np.array([0.08128174888924812, 0.08288009268103909, 0.08387401354945846, 0.08267141338584605,
                                  0.08226164758671702]),
                    50: np.array([0.07879223183975781, 0.0827252977108959, 0.0819560989708094, 0.08010382529722103,
                                  0.0812820550501859]),
                    60: np.array([0.07953559059669542, 0.07992943602705972, 0.07914554156195963, 0.07958622961580476,
                                  0.08055400434013746]),
                    70: np.array([0.07892645279488196, 0.07785366486888963, 0.07715500560886838, 0.0782535110536345,
                                  0.07744389906976061]),
                    80: np.array([0.0751391195301287, 0.07596942799339672, 0.07477356337041555, 0.07682361700981184,
                                  0.07500453118187919]),
                    90: np.array([0.0720610999260315, 0.06911399473893044, 0.07021507193557394, 0.07231888743564498,
                                  0.07126740831092235])
                },
            "feat prop 20":
                {
                    10: np.array([0.08587110134661827, 0.0850469161021059, 0.0870550256930259, 0.08395637084172214,
                                  0.08476696254059693]),
                    20: np.array([0.08353111329914127, 0.08662456341450273, 0.0853365443492488, 0.08649432255156975,
                                  0.08384113186474056]),
                    30: np.array([0.08305491057651329, 0.08345579770844662, 0.08584201605752886, 0.08347422859690116,
                                  0.08406860944151347]),
                    40: np.array([0.08242495383093058, 0.08209876996781636, 0.08269284465149088, 0.0815080018222699,
                                  0.08144970877971597]),
                    50: np.array([0.08026547827237057, 0.08105978220935531, 0.08009537525533822, 0.08056073988076867,
                                  0.08030050308365295]),
                    60: np.array([0.07991106637079275, 0.08061572638519456, 0.07977439612816631, 0.07953295761263048,
                                  0.08040276083687255]),
                    70: np.array([0.0774622687260276, 0.07794539068584948, 0.0762874679755659, 0.07755668875923993,
                                  0.07712928809009457]),
                    80: np.array([0.07527150351962614, 0.07532342841467417, 0.07568775993063619, 0.07717692673201366,
                                  0.07678210158664844]),
                    90: np.array([0.07201395114161291, 0.0716967684100695, 0.07214008944797959, 0.07294455792809872,
                                  0.07367885432127794])
                }
        },
    "toys":
        {
            "zeros":
                {
                    10: np.array([0.09118886111771915, 0.09011742721576325, 0.091879061706101, 0.09108430735410766,
                                  0.09140477195165021]),
                    20: np.array([0.08975997397493689, 0.08862616373604697, 0.09037512720929501, 0.08966933971890023,
                                  0.08933131730995511]),
                    30: np.array([0.09025739826705935, 0.08997038645434503, 0.08988013105902322, 0.08973465235007726,
                                  0.08954166313379446]),
                    40: np.array([0.09033087038234419, 0.09035468437013948, 0.09010658580455753, 0.08975236562322499,
                                  0.09101451012678553]),
                    50: np.array([0.09058148977157036, 0.08911346480426308, 0.08947094274018293, 0.09078588057490505,
                                  0.092015164079847]),
                    60: np.array([0.09200611644822705, 0.0901264504206304, 0.08939790500067528, 0.09071876858159009,
                                  0.08941945397989022]),
                    70: np.array([0.09457910183798782, 0.08931386056401615, 0.09083325325441848, 0.08957750843531818,
                                  0.0908687400090472]),
                    80: np.array([0.09186790823910128, 0.09272699944903574, 0.09233174045486194, 0.09323318555939,
                                  0.0915521917163692]),
                    90: np.array([0.0932872805856291, 0.09344520117091451, 0.09460488965582267, 0.09366904932022974,
                                  0.09463067747365753])
                },
            "mean":
                {
                    10: np.array([0.11844771487494603, 0.11821089035535837, 0.11614503174738974, 0.11915133887166297,
                                  0.11820934514139961]),
                    20: np.array([0.11165704469503444, 0.11376482146825635, 0.11393609757090734, 0.11183802910582316,
                                  0.11150256428159955]),
                    30: np.array([0.1075596386000832, 0.10889906560406071, 0.10952836671947895, 0.10814514943750178,
                                  0.10738629173831941]),
                    40: np.array([0.10507867274212906, 0.10141739161906344, 0.10255953097609996, 0.10212918173274611,
                                  0.1020177747069699]),
                    50: np.array([0.09904964117960015, 0.09877557807749776, 0.09877378767011664, 0.10033511097901028,
                                  0.09978063028259616]),
                    60: np.array([0.09477352109425526, 0.09549331141825299, 0.09562209965322828, 0.09604220776510872,
                                  0.09496639179440364]),
                    70: np.array([0.0950055843892061, 0.09312131359265591, 0.09246439107454915, 0.0930376177271216,
                                  0.09150009904765422]),
                    80: np.array([0.09187787253943683, 0.09280504340581852, 0.09032996900679467, 0.09228914367932776,
                                  0.09248127098984979]),
                    90: np.array([0.09185679604384703, 0.09184832347619143, 0.09232537215576134, 0.094189049387123,
                                  0.09234177539628342])
                },
            "random":
                {
                    10: np.array([0.12053229331657536, 0.1199669523012949, 0.11855116673150254, 0.12081518298303844,
                                  0.12047252687826153]),
                    20: np.array([0.11222895756882247, 0.11572152261941661, 0.11501814679001204, 0.11279003067704894,
                                  0.11323128184198585]),
                    30: np.array([0.10870760295677423, 0.11063422631352052, 0.11109276387762392, 0.10807904175538055,
                                  0.10888665490315728]),
                    40: np.array([0.10559999753746112, 0.10327938684737094, 0.10392915161133623, 0.10360677157071994,
                                  0.10279957508583208]),
                    50: np.array([0.09858749447985238, 0.09797888701656396, 0.09847633479791505, 0.09919761416705933,
                                  0.10108911505273871]),
                    60: np.array([0.0948574018065869, 0.09582887487117467, 0.09398504955716915, 0.096923791240032,
                                  0.09342462559203546]),
                    70: np.array([0.09211361314968892, 0.09007357707822891, 0.09006929399782654, 0.09009450320359783,
                                  0.08928660854016648]),
                    80: np.array([0.08665373732922095, 0.08687690250759497, 0.08607665187936425, 0.0884684946380979,
                                  0.08775849204250095]),
                    90: np.array([0.08577613236369785, 0.08408595454847449, 0.0864912523949055, 0.08631397651311631,
                                  0.08569546280612753])
                },
            "feat prop 1":
                {
                    10: np.array([0.114226185855761, 0.1152128615630516, 0.11476956736770955, 0.11563522846998596,
                                  0.11639398451521994]),
                    20: np.array([0.11267175603511577, 0.11304073000802903, 0.1129806037140254, 0.11156317140534713,
                                  0.11236658888633333]),
                    30: np.array([0.10991559199318447, 0.1106478005526185, 0.11002461056217337, 0.10951828377287269,
                                  0.11103917353371535]),
                    40: np.array([0.10807094419621248, 0.10786582335755211, 0.10821261501128902, 0.10869002199600508,
                                  0.10663373643018921]),
                    50: np.array([0.10473738360055375, 0.10341736230017345, 0.10616619173195158, 0.10578593167000874,
                                  0.10524722286762278]),
                    60: np.array([0.10238514185801906, 0.09859653460594876, 0.10019148008645748, 0.10107676045790058,
                                  0.09939571229240528]),
                    70: np.array([0.09798461586750803, 0.09360444486086693, 0.09845762483927097, 0.09563077868864908,
                                  0.09464621156392472]),
                    80: np.array([0.09244259947420336, 0.09291795007117386, 0.09269364338437217, 0.09253101539687036,
                                  0.09297718798628367]),
                    90: np.array([0.09026869258957913, 0.08988802204679583, 0.0904194180123207, 0.09001163640217426,
                                  0.09051392642107636])
                },
            "feat prop 2":
                {
                    10: np.array([0.11343285749576155, 0.11560102856939297, 0.11518723883558372, 0.11579232968335058,
                                  0.11580835012078931]),
                    20: np.array([0.11076079070310894, 0.11281721463964836, 0.11373643014531891, 0.11251632464012125,
                                  0.11235664032683357]),
                    30: np.array([0.11034206559434306, 0.10999265487241147, 0.11051045313965734, 0.10849362360061075,
                                  0.11085774942403256]),
                    40: np.array([0.10896868977775657, 0.10696702763411307, 0.10850215271698936, 0.1071319894117277,
                                  0.1056456150184799]),
                    50: np.array([0.10619828092526452, 0.10256603204956323, 0.10709543828736515, 0.10569113824926349,
                                  0.10550301562620329]),
                    60: np.array([0.10221401696662105, 0.10004891943062642, 0.09999900540335485, 0.10216122138672025,
                                  0.10097390720693104]),
                    70: np.array([0.09728124955679837, 0.0997663650492455, 0.09773189899940102, 0.09616643115037012,
                                  0.0962044220090921]),
                    80: np.array([0.0905188353938608, 0.08943571028803024, 0.09313282269783556, 0.08817916012202998,
                                  0.09144579353816605]),
                    90: np.array([0.07737917149921321, 0.08007672272494945, 0.08240664616226885, 0.08290066371779989,
                                  0.08007919085812004])
                },
            "feat prop 3":
                {
                    10: np.array([0.11366208013535635, 0.11577210379075445, 0.1151813783257003, 0.11545603796147941,
                                  0.11622972325027014]),
                    20: np.array([0.1117379938970521, 0.11272970760856858, 0.11376970453406274, 0.11137385978555832,
                                  0.11267338549372513]),
                    30: np.array([0.11084547035675277, 0.11138051365651978, 0.1104226989700774, 0.10892235922234654,
                                  0.10989443445553813]),
                    40: np.array([0.10770639190978025, 0.10691139103329238, 0.10789230168914814, 0.10910410879640002,
                                  0.10621596168777662]),
                    50: np.array([0.10478125436282372, 0.10222313170470185, 0.10571896748370356, 0.10528795725645923,
                                  0.10492583890519834]),
                    60: np.array([0.10283931790078087, 0.09865469798334141, 0.10012377943173467, 0.10346395275715302,
                                  0.10032226252265006]),
                    70: np.array([0.09678687040261724, 0.10009155604972125, 0.09738113007754336, 0.09666723696083564,
                                  0.096794205382972]),
                    80: np.array([0.09013184036504174, 0.08941462385909126, 0.09082094340429464, 0.08897803396164962,
                                  0.09081107022553872]),
                    90: np.array([0.0794799499151681, 0.0824223650096131, 0.08239427171273828, 0.08369502708907163,
                                  0.08313079128659487])
                },
            "feat prop 20":
                {
                    10: np.array([0.11375749506134533, 0.115442022871164, 0.11491628703308782, 0.11547318759399815,
                                  0.11528800239326227]),
                    20: np.array([0.11235631686954495, 0.11250804070595112, 0.11305890143392129, 0.11208066570836756,
                                  0.11300843061797561]),
                    30: np.array([0.11148958234257456, 0.11090536903812989, 0.11020801276219123, 0.10890533489591311,
                                  0.11146911001263203]),
                    40: np.array([0.10765407725860758, 0.10667155099214186, 0.10654792326753627, 0.10882767426610186,
                                  0.10619657149305556]),
                    50: np.array([0.10499798896447735, 0.10222225289442921, 0.10497694708024868, 0.10609743397299636,
                                  0.10382058096067069]),
                    60: np.array([0.10276658988615006, 0.10053884656381305, 0.09855072909391159, 0.10207936324276393,
                                  0.09934730776510965]),
                    70: np.array([0.09613050407616003, 0.0977313545523415, 0.09695960132636514, 0.09624348247768841,
                                  0.09413053914457725]),
                    80: np.array([0.09025019441045262, 0.09091534347627861, 0.09097960516702024, 0.09065927636911901,
                                  0.09270205883808452]),
                    90: np.array([0.08278341966554231, 0.08374238330042405, 0.08397461991739105, 0.08694782884731188,
                                  0.0824221778643861])
                }
        },
    "sports":
        {
            "zeros":
                {
                    10: np.array([0.08336141226871187, 0.08230090329444176, 0.08326913870705054, 0.08230549912464055,
                                  0.08390104304953003]),
                    20: np.array([0.08345473439357387, 0.08263673354678408, 0.08330886427457343, 0.08320687361695615,
                                  0.08369604663587776]),
                    30: np.array([0.0823586165822747, 0.08242034910594183, 0.08240571384084275, 0.08235645483422335,
                                  0.08118938052803447]),
                    40: np.array([0.08187406998823415, 0.08203695424202748, 0.08215209540540612, 0.08354481043448479,
                                  0.08378757627925948]),
                    50: np.array([0.08156635015361263, 0.08176367082307201, 0.08252670134027691, 0.08391443011668843,
                                  0.08419206395743405]),
                    60: np.array([0.08314612331869704, 0.08325422456875978, 0.08192218119126186, 0.08285418655276629,
                                  0.0822991285069847]),
                    70: np.array([0.08291337459896733, 0.08378593650489242, 0.0838696430821055, 0.08314779655784708,
                                  0.08423156471026089]),
                    80: np.array([0.08452078967481053, 0.08428384785623606, 0.08364008582564658, 0.08285927319978244,
                                  0.08234227692115864]),
                    90: np.array([0.0840996130708881, 0.08347434591037048, 0.08383711415907051, 0.08405142378336672,
                                  0.08367158734137814])
                },
            "mean":
                {
                    10: np.array([0.0948158953773525, 0.09619413131031941, 0.09425920654408913, 0.09521556412677494,
                                  0.09641826454170514]),
                    20: np.array([0.0946438630687879, 0.0915505151175381, 0.0929082907451214, 0.09268209055970555,
                                  0.09311123279606655]),
                    30: np.array([0.09131854853657682, 0.09162684541746255, 0.08906277272984457, 0.09178916248608758,
                                  0.08984331819962937]),
                    40: np.array([0.08809471579607847, 0.08880376877226087, 0.08744216252852936, 0.0893239999799081,
                                  0.09042282886094134]),
                    50: np.array([0.0872738123601792, 0.08643327855950209, 0.08627289300950615, 0.08500616173349843,
                                  0.08752365158305277]),
                    60: np.array([0.08582925153522043, 0.08565743103040781, 0.08400848731281844, 0.08606529980562351,
                                  0.08658447244910043]),
                    70: np.array([0.08631475745304035, 0.08405191516236875, 0.08463674399395082, 0.08562327579584951,
                                  0.08332512872129659]),
                    80: np.array([0.0817807255535168, 0.08311977594173231, 0.08381505142828795, 0.0841029137755195,
                                  0.08318487438345412]),
                    90: np.array([0.0835061374542214, 0.0824432724822975, 0.08324266921765482, 0.08409479298817792,
                                  0.0835681242333088])
                },
            "random":
                {
                    10: np.array([0.0958812344353024, 0.09738879000689737, 0.09563775481503073, 0.09614910886829742,
                                  0.09659981098948535]),
                    20: np.array([0.09478725850998898, 0.09354215880676992, 0.09246689141129676, 0.09327936885274865,
                                  0.0946212765083046]),
                    30: np.array([0.09034216771885509, 0.09301969213763085, 0.08946948139258293, 0.0921345396403701,
                                  0.09287368875127175]),
                    40: np.array([0.08888511935182013, 0.09039662148948183, 0.08821925112469542, 0.0894428568881579,
                                  0.0903209268805607]),
                    50: np.array([0.08551914339212018, 0.08389196465455181, 0.08669348955718989, 0.08402109238108213,
                                  0.08657724461815691]),
                    60: np.array([0.08298435340371238, 0.08422263789802654, 0.08477890123818087, 0.0832226689945326,
                                  0.08432324706075586]),
                    70: np.array([0.08158203685201317, 0.08164520720739134, 0.08085620917253572, 0.08211822960719385,
                                  0.08212983188592556]),
                    80: np.array([0.07832613541061498, 0.08031729713004103, 0.08130248320975082, 0.08070900528516803,
                                  0.0794900617192857]),
                    90: np.array([0.0771257196174028, 0.07982691109410449, 0.08005883032512806, 0.07910599815148468,
                                  0.08051319112271706])
                },
            "feat prop 1":
                {
                    10: np.array([0.09607291301927835, 0.09632030354320267, 0.09585483418338875, 0.09519687962293276,
                                  0.09436916297401869]),
                    20: np.array([0.09561625042240486, 0.095587268766368, 0.09594322250822154, 0.09378848377552874,
                                  0.09462313591935088]),
                    30: np.array([0.09536175889263576, 0.09451107365588204, 0.09449313111746738, 0.09342632789389235,
                                  0.09394770634294657]),
                    40: np.array([0.09293589862891251, 0.09314652597312055, 0.09286170548885665, 0.09363631112239647,
                                  0.09231063419862195]),
                    50: np.array([0.09031173536003681, 0.0917155999020563, 0.09089587717234701, 0.09115180467781069,
                                  0.09056405982386406]),
                    60: np.array([0.09111009755722185, 0.09077199883050435, 0.08928278367613662, 0.08968133910083474,
                                  0.09275275874261864]),
                    70: np.array([0.08474642039673068, 0.08460871281468171, 0.08327959087427451, 0.08451547992924104,
                                  0.0839190036370319]),
                    80: np.array([0.08324129600759375, 0.08466162063660623, 0.08449787745338252, 0.08541079673364875,
                                  0.08411744980022762]),
                    90: np.array([0.08318480860784618, 0.08330908565698406, 0.08418802874371935, 0.08358850544011441,
                                  0.08317819545188235])
                },
            "feat prop 2":
                {
                    10: np.array([0.09521300080132762, 0.09634616181949278, 0.09461690377665916, 0.09534140286578623,
                                  0.09520123018459656]),
                    20: np.array([0.09522982127829138, 0.0948871072816205, 0.09669137336600302, 0.09491939093612052,
                                  0.09504200243916192]),
                    30: np.array([0.09512341270713327, 0.09442715800514406, 0.0950109622995235, 0.09502261793469242,
                                  0.09493991484906973]),
                    40: np.array([0.09340492931034163, 0.09326857203672508, 0.09272320692102767, 0.09420619948177367,
                                  0.09216679983105676]),
                    50: np.array([0.09040278187965477, 0.09078442772784145, 0.0916785046191354, 0.09047402724870576,
                                  0.09191171181526885]),
                    60: np.array([0.09166635459499001, 0.09228767291111349, 0.0904292982966772, 0.09122913993530948,
                                  0.09176543437867088]),
                    70: np.array([0.08926208726125558, 0.0884234597992515, 0.08989981906222072, 0.08974064596705501,
                                  0.088622813535647]),
                    80: np.array([0.08311005884577632, 0.0859362913510121, 0.08588587738034349, 0.08448296331509178,
                                  0.08503464970261719]),
                    90: np.array([0.07913856281763006, 0.08126250279503429, 0.08069118201867227, 0.08043724952911825,
                                  0.08092363784018189])
                },
            "feat prop 3":
                {
                    10: np.array([0.09432464582574213, 0.09654956077057263, 0.09349948120747406, 0.09517391162686643,
                                  0.09540206465627106]),
                    20: np.array([0.09533838117422021, 0.09490253570054198, 0.09564604408024535, 0.09523555505098594,
                                  0.09495087115608107]),
                    30: np.array([0.09545650898811285, 0.09389615942219742, 0.09447805737790382, 0.09492973927671003,
                                  0.09526030668152094]),
                    40: np.array([0.09215964952242221, 0.09416157977110572, 0.09157046683676458, 0.0946616997981274,
                                  0.09334177067207583]),
                    50: np.array([0.09077387285835241, 0.09163840780260077, 0.09123883486128384, 0.09042709962196879,
                                  0.09284618055977129]),
                    60: np.array([0.0917954428502378, 0.09105632737358155, 0.0908901301062711, 0.09024527646903927,
                                  0.0904494317871581]),
                    70: np.array([0.08844068416152737, 0.08950913716043905, 0.09033809788639934, 0.08736526276510923,
                                  0.08746517745222245]),
                    80: np.array([0.0843531105182631, 0.08687840594708356, 0.08656967560028785, 0.08514897711846083,
                                  0.08564331833076635]),
                    90: np.array([0.08085428937101895, 0.08280100157379283, 0.08078693360984962, 0.08206602226738495,
                                  0.08251488883406229])
                },
            "feat prop 20":
                {
                    10: np.array([0.09492580603361336, 0.09639495347310818, 0.09392746116331532, 0.09520670711420738,
                                  0.0941124200048773]),
                    20: np.array([0.09525404992105779, 0.0948191414613036, 0.09556523606808297, 0.09527772067756714,
                                  0.09397169160547296]),
                    30: np.array([0.0946780388970588, 0.09406326023864889, 0.09414009422646107, 0.0939652765079801,
                                  0.09569476880429476]),
                    40: np.array([0.09346781963857007, 0.09409297696595374, 0.09249907994630022, 0.09505177646371427,
                                  0.09257403449153817]),
                    50: np.array([0.09072659827489972, 0.09151080543106005, 0.0906747049892082, 0.09065995989097518,
                                  0.0923561770380592]),
                    60: np.array([0.0906534213872196, 0.0902049375203684, 0.09003466870445949, 0.0897593806909357,
                                  0.09075435288889322]),
                    70: np.array([0.08891039757187157, 0.08849342578390623, 0.09025969762247031, 0.08808070520692873,
                                  0.08675593599622773]),
                    80: np.array([0.08516988376082484, 0.08578380635990514, 0.08652353710331448, 0.08600678364543586,
                                  0.08632299922844354]),
                    90: np.array([0.08245508498851215, 0.08424891062278304, 0.08259246020105851, 0.08164251599972283,
                                  0.08362875126517887])
                }
        }
}

mean_std = {
    "zeros": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["zeros"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["zeros"].items()])
    },
    "mean": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["mean"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["mean"].items()])
    },
    "random": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["random"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["random"].items()])
    },
    "feat prop 1": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["feat prop 1"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["feat prop 1"].items()])
    },
    "feat prop 2": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["feat prop 2"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["feat prop 2"].items()])
    },
    "feat prop 3": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["feat prop 3"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["feat prop 3"].items()])
    },
    "feat prop 20": {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array([np.mean(value) for _, value in values[dataset]["feat prop 20"].items()]),
        "std": np.array([np.std(value) for _, value in values[dataset]["feat prop 20"].items()])
    }
}

if method == 'feat prop':
    feat_prop_np = np.empty((len(layers), 9, 5))
    for fp, v in enumerate(layers):
        for idx in range(9):
            feat_prop_np[fp, idx] = values[dataset][f'feat prop {v}'][(idx + 1) * 10]

    feat_prop_np_indices = np.argmax(feat_prop_np, axis=0)
    feat_prop_np = np.amax(feat_prop_np, axis=0)
    mean_std['feat prop'] = {
        "x": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
        "y": np.array(
            [np.mean(feat_prop_np[row_idx]) if feat_prop_np[row_idx].shape[0] > 1 else feat_prop_np[row_idx] for row_idx
             in range(9)]),
        "std": np.array(
            [np.std(feat_prop_np[row_idx]) if feat_prop_np[row_idx].shape[0] > 1 else 0.0 for row_idx in range(9)])
    }

fig = plt.subplots(figsize=(10, 7))
# plt.errorbar(mean_std["feat prop_1_norm"]["x"], mean_std["feat prop_1_norm"]["y"], mean_std["feat prop_1_norm"]["std"])
plt.bar(mean_std[method]["x"], mean_std[method]["y"], yerr=mean_std[method]["std"], width=3)
tikzplotlib_fix_ncols(fig[0])
tikzplotlib_fix_ncols(fig[1])

tikzplotlib.save(f'./data/{dataset}/rq1.tex')
