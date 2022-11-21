from readData import getData
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import pandas as pd
import statistics
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px

'*********** Load and Split Data ***********'
df_train, df_val, df_test = getData()
#y_train = df_train[['data_type']]
#X_train = df_train.drop(['data_type'], axis=1)
#y_val = df_val[['data_type']]
#X_val =  df_val.drop(['data_type'], axis=1)
#y_test = df_test[['data_type']]
#X_test = df_test.drop(['data_type'], axis=1)
'******* Factor analysis procedure *********'

signal = df_train.loc[df_train['data_type'] == 1]
backgrond = df_train.loc[df_train['data_type'] == 0]

df_train_short = pd.concat([signal.iloc[:1000, :], backgrond.iloc[:9000, :]])
y_train_short = df_train_short[['data_type']]
df_train_short = df_train_short.drop(['data_type'], axis=1)


signal_test = df_test.loc[df_test['data_type'] == 1]
backgrond_test = df_test.loc[df_test['data_type'] == 0]

df_test_short = pd.concat([signal_test.iloc[:100, :], backgrond_test.iloc[:900, :]])
y_test_short = df_test_short[['data_type']]
df_test_short = df_test_short.drop(['data_type'], axis=1)

print(y_train_short.value_counts())
print(y_test_short.value_counts())

# skip section 2.2.1 factor analysis procedure 

'********* Noise Corruption *************'
# Gaussian noise 
'''features = np.array([[1,1,1,1,1],[4,4,5,4,3],[5,5,4,3,5]])
labels = [1, 0, 0]

X_test = np.array([[2,2,2,2,3], [4,4,3,4,3]])
y_test = np.array([[1], [0]])'''

dampenngFactors = np.linspace(0, 1, 11)

noise_list = []

def addNoiseList(features):
    for feature in features:
        feature_list = []
        for factor in dampenngFactors: 
            sd = np.std(feature)
            q = factor * sd 
            noise = np.random.normal(0, q, len(feature))
            a = feature + noise
            feature_list.append(a)
        noise_list.append(feature_list)


def addNoiseDf(df):
    for factor in dampenngFactors:
        df_temp = df.copy()
        for (name, feature) in df.items():
            new_feature = addNoiseColumn(feature, factor)
            df_temp[name] = new_feature
        noise_list.append(df_temp)

def addNoiseColumn(feature, factor):
    sd = np.std(feature)
    q = factor * sd 
    noise = np.random.normal(0, q, len(feature))
    a = feature + noise
    return a


addNoiseDf(df_train_short)
feature_names = df_train_short.columns

'********************************************'
models = []

models.append(RandomForestClassifier())
models.append(SVC())

def train(model):
    average_parameter_value = []
    feature_variance = []
    acc_list = []
    i = 0 
    for data in noise_list:
        av = []
        print(str(i))
        model.fit(data, y_train_short.values.ravel())
        print(model.feature_importances_)
        pred = model.predict(df_test_short)
        acc_list.append(accuracy_score(pred, y_test_short))
        i += 1
        feature_variance.append(data.var().tolist())
        for feature in feature_names:
            av.append(data[feature].mean())
        average_parameter_value.append(av)
    return acc_list, average_parameter_value, feature_variance
    
print('Model = RandomForestclassifier')
acc_0, average_parameter_value_0, feature_variance_0 = train(models[0])
#print(acc_0)
#print(feature_variance_0)

print('***************')

#print('Model = SVC')
#acc_1, average_parameter_value_1, feature_variance_1 = train(models[1])
#print(acc_1)
#print(feature_variance_1)

'''acc_0 = [0.968, 0.962, 0.957, 0.96, 0.956, 0.953, 0.954, 0.952, 0.95, 0.951, 0.948]
acc_1 = [0.953, 0.955, 0.955, 0.954, 0.954, 0.956, 0.953, 0.954, 0.954, 0.954, 0.954]

apv_0 = [[0.06395540065262467, 0.041908727896117486, 0.49722984471783277, 0.07488109269812702, 0.08484346025586129, 0.035011500570937014, 0.1121, 0.8543154844939709, 0.36868893554508686, 0.2557, 0.5016280389919906, 0.021477548852516338, 0.028026881773769857, 0.5153, 0.1604278384980973, 0.1869780879214406, 0.12724166666666667, 0.035370329612001934, 0.04536668829941627, 0.13018469353169204, 0.5079167579951646, 0.16824213688839418, 0.08585333333333334, 0.03486874718144536, 0.04143529430724848, 0.034678917789915796, 0.5018537025002413, 0.02745748920526039, 0.49729986714010593, 0.16484905523210763, 0.077919764656201, 0.05026590885287151, 0.49865560834166206, 0.502, 0.028338026073910942, 0.20643494631052017, 0.03356335403171638, 0.5028958681980563, 0.2858, 0.25250794654637576, 0.09028545929044485, 0.495184203575803], [0.06390546737817031, 0.0418084062382037, 0.49750861610294694, 0.07483853106123071, 0.08491059266190627, 0.03508717827741764, 0.1121295933992164, 0.8539351725950807, 0.36869212137914276, 0.25584952487491547, 0.5013015392755736, 0.021475049246785862, 0.028039077511392542, 0.51498649719001, 0.16032822526960805, 0.18693085620410177, 0.12719200642137665, 0.035368118993502076, 0.04536367748357029, 0.13018927860187088, 0.5078629616282989, 0.16841740366131616, 0.0861424567902407, 0.03482229592567872, 0.04143496135472017, 0.034723058949145874, 0.5019085180777851, 0.027467887296069166, 0.49742050403421, 0.16493016388254259, 0.07793210530546726, 0.05023805034319565, 0.4984546974318061, 0.5016534356805714, 0.028300437564193662, 0.20643487696009505, 0.03353836235999614, 0.5027757299429827, 0.2856064242569159, 0.25241398122531894, 0.09031390556194249, 0.49517834729359755], [0.0641263933845627, 0.04190695314516164, 0.498489081911673, 0.07491040815852983, 0.08482486975845784, 0.0350570064076475, 0.11184937010072524, 0.854131289147516, 0.3686010848515237, 0.25557856763023556, 0.5017498380812753, 0.02144937147404033, 0.0280368013960686, 0.5148803676641863, 0.16017514411455927, 0.18692384497360895, 0.12709791282425237, 0.03549220316020621, 0.04537264707146679, 0.13020664412623736, 0.5076458115850113, 0.16780045174984753, 0.0858063350907868, 0.03485227229160197, 0.04166830076219542, 0.03457333135865792, 0.5030078006497932, 0.02751465769016961, 0.4977376401521604, 0.16483553441781357, 0.07786833321143025, 0.050264228260964466, 0.497747127863144, 0.5009533064795408, 0.02844240696546213, 0.2063479674587114, 0.033737126990180936, 0.5025223421771075, 0.2855042448701955, 0.25254594047101, 0.09031126912679509, 0.49507123861978153], [0.06406492111170946, 0.04198244298805221, 0.49735090153367034, 0.07494758719592637, 0.08490380997023586, 0.0351500441912402, 0.11184442578883189, 0.8551878053131033, 0.3685878439249444, 0.25704542165965905, 0.5033145200390969, 0.021479951899453766, 0.02803777253943234, 0.5127090486634336, 0.15944233234249777, 0.18712518412022558, 0.12797409678188756, 0.03528313442619896, 0.04543628909672584, 0.1303120796387254, 0.5091095814658587, 0.16954019477327958, 0.08608823793029571, 0.03489687193110164, 0.04123344118596016, 0.03467862803547467, 0.5008055836688656, 0.027689911091891013, 0.4969018532920788, 0.1650416147417201, 0.07790643032410653, 0.0501953354320448, 0.4980246951817591, 0.5038005524191581, 0.028440131402159546, 0.20651811718007548, 0.03336827939521813, 0.5024679856354546, 0.28631551788193693, 0.25247375639988723, 0.09031350118429016, 0.4959525407971665], [0.06379473712909293, 0.041922428805985026, 0.4973933320963119, 0.07484473787822331, 0.08492337529547699, 0.034801329013821165, 0.11175021680961689, 0.8536552786886281, 0.36845139153247447, 0.2556354581344521, 0.5014997155064708, 0.02136620771929493, 0.027957827509555312, 0.5173139753637558, 0.16059081614832907, 0.18730008797196807, 0.12801870816745364, 0.035194885950189665, 0.04556547682850954, 0.13013204025105596, 0.5060687015064066, 0.1698939800566325, 0.08525795700045001, 0.034800085297181683, 0.041767130204198075, 0.03485359237398535, 0.4997826288952484, 0.027449775429961288, 0.496795665532165, 0.16505406428295924, 0.0776969889738329, 0.050383877574469944, 0.4994026281481998, 0.49944382951716054, 0.028435934342822725, 0.20649579945046367, 0.03353365720750878, 0.5035589403055312, 0.2852612454672161, 0.25268321700076846, 0.09018357878419736, 0.4943802431791239], [0.0640449453188408, 0.04210909224011532, 0.49622864126323, 0.07511853789671277, 0.08478826233556129, 0.034568528506318946, 0.11282345664509265, 0.8543141065402575, 0.3683825033444686, 0.2582583502361505, 0.5019673109611269, 0.021409756775155192, 0.028010533626837163, 0.5154054178682679, 0.16073871989748303, 0.1863966664213702, 0.12735219963765487, 0.03520923462951035, 0.045112907062890124, 0.13003097297479574, 0.5075978883120964, 0.16372881192904, 0.08733294272688032, 0.0350186060558769, 0.04163630796178963, 0.034258084132525554, 0.5004323898696689, 0.027461531876165304, 0.4967940422767561, 0.16459385371973473, 0.07777258224632762, 0.04937062819511746, 0.4993780673285083, 0.5041629933696052, 0.02802872057478471, 0.20589270325087142, 0.033536275427643945, 0.5018853414222841, 0.28600129651756634, 0.2524768489151071, 0.0899888684192206, 0.4954809027557383], [0.06436961739782032, 0.04202216778340812, 0.4966584103859819, 0.07485898756362543, 0.08484045488589291, 0.03581367268901794, 0.11263795203396298, 0.8552770418306118, 0.36810184608055907, 0.2528166028759721, 0.5047410322547099, 0.0214359971712937, 0.028153611575855796, 0.5152905510688548, 0.16227236404360254, 0.18734593985908082, 0.12733483601581624, 0.03560010720719615, 0.04586133478001273, 0.13039767354197745, 0.5073855488165713, 0.1719112544586296, 0.08694072256305516, 0.034863518770811786, 0.041182883854549567, 0.03489231115958294, 0.501316137622654, 0.027556892615771527, 0.4959776087979889, 0.16473087404075032, 0.07800417988769892, 0.05044022992229064, 0.49842211710588347, 0.49742550470059066, 0.028250057101149827, 0.20627351222734483, 0.033450448753108125, 0.5018064633660202, 0.286504179098442, 0.25273439983141255, 0.09031578142973369, 0.49637280223236296], [0.06408125035078395, 0.04132199692213761, 0.49741202748856567, 0.07462539008213002, 0.08517389000767772, 0.035696869977318336, 0.11109760542129302, 0.8549618751416077, 0.36837827827019426, 0.2599748983935715, 0.5047822588341827, 0.02158432196042182, 0.02813718253950194, 0.5149089475115319, 0.15828395902340273, 0.1869191456765489, 0.1272075902976302, 0.03512465153655411, 0.04560265632030738, 0.13019082758832629, 0.5076650857576711, 0.16497486858583146, 0.08573951609502199, 0.034770037065410125, 0.04183270404498017, 0.03453859509620235, 0.5033660479333835, 0.026790154715766852, 0.494411406562708, 0.16440663685640194, 0.07821342771146453, 0.050387424958225784, 0.4966208511427203, 0.5034910609761252, 0.028368181634077404, 0.20607365576446537, 0.033465467991699825, 0.5006240741978643, 0.28133814957817005, 0.25265651299429737, 0.09037508705324095, 0.4982081492366292], [0.06360485804233985, 0.04266933508500689, 0.49564122915237785, 0.07467958646674569, 0.08496222995640583, 0.03517127467031175, 0.11147403821994019, 0.8553021282852081, 0.3693391789107428, 0.25833465758002694, 0.5016927321400267, 0.021613267477897498, 0.028194033799070404, 0.5114794651322682, 0.15749251589970137, 0.18670815357580392, 0.12431377611178467, 0.03528035768870764, 0.044717850245099916, 0.1302754151332448, 0.5065383906835011, 0.1677932519042302, 0.08595198198549374, 0.034793106310516866, 0.04159189660188872, 0.03450545770174321, 0.5016637205446915, 0.02706458523825257, 0.4949263658853848, 0.16446940965577772, 0.07797811710040557, 0.05055175333053711, 0.4979487697447189, 0.49721813170502116, 0.028133646993684737, 0.20614936994455438, 0.03347763443694946, 0.506686919556377, 0.28006069352488017, 0.2528368044356627, 0.09040931993420141, 0.49449660219951164], [0.0635671849492694, 0.0419182245782636, 0.49748952328192736, 0.07513675980025951, 0.08512446866242591, 0.03415855587692026, 0.11281555175840904, 0.8549230301585109, 0.36957641911663697, 0.2552980904313074, 0.5028788390381148, 0.02129811389273186, 0.02794105521039541, 0.5204278505132858, 0.16285835988101016, 0.18683325130011383, 0.12633652910232737, 0.03565600302389893, 0.04626057542790004, 0.12972706255982253, 0.5042204628259875, 0.16471743370437786, 0.08418571944824471, 0.03530120145729457, 0.04107221887271444, 0.03520766615128895, 0.5001888260779167, 0.02724414233862413, 0.4991531996372021, 0.16523002599155356, 0.07834188521056662, 0.05069733112253647, 0.4991994421474983, 0.5050573976542734, 0.028432130170075794, 0.20737286484052483, 0.03411689723915245, 0.5007507813255946, 0.28108546999411926, 0.2523696777921435, 0.09061783281850992, 0.4994840363457693], [0.06411769462073631, 0.04178525774699548, 0.49598949594219877, 0.07455398639815396, 0.08521716086571954, 0.03494112998220759, 0.11197762476747021, 0.8560440619198791, 0.3689509044076184, 0.24997979211870502, 0.5019547585708692, 0.02147735175381361, 0.02808017410294003, 0.5218570438233595, 0.1645904386604769, 0.18709614010307676, 0.12808618119283594, 0.035512824113283874, 0.04472714812722341, 0.12971461090480602, 0.5045087286362484, 0.17470826640154247, 0.08454333587415568, 0.03491633469644961, 0.041331892750471395, 0.03470929579220701, 0.5024269782338845, 0.027469890485679584, 0.5003569926042523, 0.16577754454650914, 0.0781802451464131, 0.05099786644645805, 0.4973364095310897, 0.4973724967829136, 0.02913800817250742, 0.20637304082968658, 0.03354333740746658, 0.502744331209197, 0.28302745698977233, 0.25220363313663186, 0.0901032887795092, 0.4937156475282671]]
apv_1 = [[0.028338026073910942, 0.5028958681980563, 0.04143529430724848, 0.20643494631052017, 0.8543154844939709, 0.5016280389919906, 0.49729986714010593, 0.25250794654637576, 0.2557, 0.09028545929044485, 0.08585333333333334, 0.1604278384980973, 0.5153, 0.035011500570937014, 0.07488109269812702, 0.1869780879214406, 0.077919764656201, 0.12724166666666667, 0.16824213688839418, 0.5079167579951646, 0.06395540065262467, 0.2858, 0.49865560834166206, 0.495184203575803, 0.36868893554508686, 0.13018469353169204, 0.028026881773769857, 0.03486874718144536, 0.041908727896117486, 0.03356335403171638, 0.035370329612001934, 0.034678917789915796, 0.16484905523210763, 0.04536668829941627, 0.05026590885287151, 0.5018537025002413, 0.502, 0.08484346025586129, 0.02745748920526039, 0.021477548852516338, 0.49722984471783277, 0.1121], [0.028298109626808923, 0.5031085959381657, 0.04148219421066854, 0.20639253989873432, 0.854357093728406, 0.5024827362433263, 0.4974735737255938, 0.2524805074512214, 0.2555506107548271, 0.09027179712783703, 0.0861514899502213, 0.16009075181681973, 0.5156704960361436, 0.03510054572979976, 0.07486856435136814, 0.18699919498229223, 0.07790066994561708, 0.1272190966470865, 0.16873559834269491, 0.5076377865386947, 0.06404566547181065, 0.28585035468952974, 0.4986052724946403, 0.4953929893284844, 0.36859487030240595, 0.13022420545499788, 0.028020470463479746, 0.034886055224975294, 0.041986041406482474, 0.03356441224856907, 0.035350078041528024, 0.03466201789631729, 0.1648903584266923, 0.045422596246512674, 0.050302123600791795, 0.5026623841743242, 0.501905273027545, 0.08484034751948481, 0.027441921440034673, 0.021486169472204407, 0.49720685458727093, 0.11196880288576667], [0.02837739050186184, 0.5024611259610907, 0.04155272558324325, 0.20644171169355646, 0.8542074042548902, 0.5014727839101812, 0.4970200725581082, 0.25254954849408284, 0.2566231160223994, 0.09029716725339454, 0.08527093570177091, 0.16013374672648112, 0.516217130367365, 0.03503288086077357, 0.07482833670491625, 0.18698125264605028, 0.0778176933285519, 0.12733307099443342, 0.16939392710805978, 0.50839706989684, 0.06391173539373843, 0.28557388569169545, 0.4987709979959176, 0.4949674946634388, 0.3685426044019956, 0.13019740668971314, 0.02802318431360063, 0.034840225632637944, 0.04173789527948159, 0.033568735079082965, 0.03535381471014001, 0.03463303485803674, 0.16476032161080997, 0.04536174762156797, 0.050264941497829556, 0.501874043463766, 0.5048554032397599, 0.0848730508390232, 0.027623850984688964, 0.021512419015350438, 0.4972666111782928, 0.11226199821450832], [0.028290295588412475, 0.5026097170947822, 0.04156990343030651, 0.20657039964606352, 0.8541116935176125, 0.5027799462262433, 0.495517368479425, 0.2525474488685933, 0.25783502278336307, 0.09019534110312893, 0.08569323254347731, 0.1605429379245568, 0.5158868697539797, 0.03504776374090677, 0.07512292335438076, 0.1870671530702511, 0.0777993014317694, 0.127551164277078, 0.16803744754910946, 0.508221726085458, 0.06413173574641075, 0.28651559804377136, 0.49824642649706247, 0.4972757581200127, 0.36834051305462806, 0.13013609081121466, 0.027990728662616742, 0.0347965695498697, 0.04197201710592874, 0.03373575637423744, 0.035457037780444865, 0.03467027816672274, 0.16521430367763998, 0.04530319302417466, 0.049835614724168434, 0.5019692235949468, 0.5005610216226534, 0.08494547125258983, 0.027563968420584523, 0.021520075941858674, 0.4968540035776117, 0.11232059057886813], [0.028639204191522275, 0.5037224908621635, 0.041810940041948956, 0.2065706543100744, 0.8525122902087845, 0.5031726613875604, 0.4989883508000075, 0.25247699337658136, 0.2567298798071558, 0.09015880649527251, 0.08503040381673287, 0.1594328119231929, 0.5132047264859585, 0.03486592328978933, 0.07504207478157816, 0.18703047440088705, 0.07792557734213947, 0.12666788423188294, 0.16681639342177476, 0.5070799374925232, 0.06373475015345355, 0.2848521112950092, 0.49824701422590906, 0.4941942945570716, 0.3689643467781867, 0.1301161978933617, 0.027989707232331865, 0.03489952918914927, 0.04180217208431519, 0.03338014458033159, 0.03556314666322483, 0.03473568992268838, 0.16457286074629618, 0.04556191003779854, 0.05050737027405537, 0.5008688308259327, 0.5026220719216968, 0.08487111054518068, 0.02729513326647972, 0.02149620217869665, 0.4973338194371774, 0.11164713396703457], [0.02863603883594478, 0.5019069186640477, 0.04152244256614645, 0.2063321950923949, 0.8552151668899495, 0.5031099177767616, 0.49786094153899246, 0.25232274471075694, 0.2554155048497967, 0.09035619944463476, 0.08538349651609514, 0.16282644554751674, 0.5155078298872487, 0.03530357773647629, 0.07500340789556831, 0.18699713946569144, 0.0779532503175858, 0.12810609110330468, 0.1661424877687225, 0.5069353825277538, 0.06392642738005218, 0.2854816122606854, 0.4994978796228972, 0.4931468722033792, 0.36839990405400347, 0.13014568928576598, 0.02806471366469937, 0.03502598611201938, 0.042072853069291885, 0.03326915752684586, 0.03514060471281377, 0.03500335838904058, 0.1646420928500966, 0.045212974543321835, 0.050080708094237675, 0.5036006417515476, 0.5034067876742012, 0.08479153524188651, 0.027785317905990627, 0.021485305401823536, 0.4973396700822607, 0.11214762700569582], [0.02781339159959315, 0.5056675382096038, 0.041650324274660894, 0.2067387684024955, 0.8536154148775701, 0.49991715798951025, 0.49757300256251163, 0.2524931463027085, 0.2541759925588193, 0.09022445112683267, 0.08531261827783174, 0.15636285102860217, 0.5145726836814408, 0.03467427670677955, 0.07476050800648341, 0.18662887020824093, 0.07707085814475631, 0.12632437572833205, 0.16753639798959666, 0.5102861408329858, 0.06388736866426083, 0.28311060172061026, 0.49762038121384067, 0.4948441828768555, 0.3687313992446968, 0.13017619710693906, 0.02811758627005296, 0.034933000783542174, 0.041640765776409995, 0.0336799027042945, 0.03584385461013066, 0.03435503326716058, 0.1653057296297894, 0.045308815088025124, 0.05030888934501972, 0.5025051812199652, 0.5042088238606836, 0.08468610504784685, 0.027354345379686566, 0.021477883353168047, 0.4979937887673148, 0.11206323964712588], [0.028388404358940352, 0.501922539877719, 0.041996562089034406, 0.2065392726254305, 0.8531363588168956, 0.5050252352159507, 0.4941215707408268, 0.25195457834265644, 0.25547208052572085, 0.09020475581834883, 0.0850214170868441, 0.158414314199169, 0.040780726910235765, 0.20647215964866006, 0.8543907754811298, 0.5037345393771844, 0.5024558579442464, 0.25209491856116245, 0.26013626871345397, 0.0899137585808616, 0.08586499220997253, 0.16733015012123445, 0.526063921007361, 0.03588620909032412, 0.07494894064303248, 0.18694684489781327, 0.07872030543022192, 0.1278075484806352, 0.16351311460867227, 0.5048044974965266, 0.06392211662772687, 0.29180847827056183, 0.5030413615468404, 0.49613374321391157, 0.37062822488004876, 0.130277864989742, 0.028063417359896238, 0.034930308970006034, 0.04084926364742935, 0.033706227573139756, 0.03554513361457212, 0.034595083041427346, 0.16423683612284345, 0.04445670310871433, 0.05135630450940644, 0.4991771242442058, 0.5035359319249292, 0.08510642103424325, 0.028398327900407735, 0.021622471084204568, 0.4968082605532335, 0.10996471750874387]]

fv_0 = [[0.004853527567298211, 0.08270415337840736, 0.0016554065197528043, 0.002652038605432648, 0.00043725012115650814, 0.1071959879406969, 0.005327365641279487, 0.06714210064453999, 0.11098500775757877, 0.0007084390677100904, 0.24979088908890887, 0.0024914923046286206, 0.030650936915913817, 0.0029335691792253915, 0.060928568926459425, 0.2041387738773878, 0.08308714952249051, 0.0008138942024771402, 0.0018738615477551436, 0.06331127275721861, 0.00156194988239834, 0.0837096374107845, 0.25002100210020994, 0.19033654365436545, 0.08287274857866926, 0.03289899375431595, 0.0014246393199960207, 0.0009865695669613857, 0.0022659471018843386, 0.006436182233108007, 0.0027341387418320508, 0.016092143658810325, 0.003955714291909548, 0.0027123451626249245, 0.00011267367021740775, 0.002092257672069732, 0.0018674436864090987, 0.01764118349334933, 0.06160582315510008, 0.0018218753667138, 0.001445962101671745, 0.0025359850845342647], [0.004894138766217711, 0.08348627005640852, 0.0016697643234361698, 0.002671474954514151, 0.00044187210878445096, 0.10801516289302224, 0.00538903306267693, 0.06781503080839119, 0.11200629022241518, 0.0007168553254916849, 0.25218588946575043, 0.0025137241715463845, 0.030900731427854586, 0.00295936248002833, 0.06164389815837776, 0.2059807861578539, 0.08409037330615662, 0.0008256256383130057, 0.0019023909263014933, 0.06384732070875687, 0.0015841796058652143, 0.08467157750134731, 0.2535166382838031, 0.19210756612643418, 0.08358249629265056, 0.0332213951833465, 0.0014397970951054533, 0.0009962198784309905, 0.0022833229451717776, 0.006488237755134504, 0.0027635592119190132, 0.016246584637311583, 0.0040022849341409435, 0.0027444649727536865, 0.00011376681177545398, 0.0021128588301181214, 0.001888631948555626, 0.01778324583761707, 0.062210802202049255, 0.0018416948739576125, 0.0014519757518074792, 0.0025700376667509768], [0.005043986199230888, 0.08571993271765715, 0.0017226966366072902, 0.0027719653633261226, 0.0004558314780519755, 0.11157736838460477, 0.005504656360814008, 0.07011694240505972, 0.11555826620093672, 0.0007330799422788359, 0.25817645732786576, 0.0025959357039422593, 0.03168425447772218, 0.0030607139512554274, 0.06363197206885646, 0.2139394999669867, 0.0863152392151513, 0.0008436745555454873, 0.001955656386819755, 0.06564432089583774, 0.0016334493108839812, 0.08690726667612438, 0.2614426861104372, 0.1970029940460095, 0.08637599943841794, 0.034222494042727224, 0.0014821006747462015, 0.0010266097823545756, 0.0023693196960222727, 0.006673193763948344, 0.0028452520396721167, 0.01678678276439021, 0.004124091855131796, 0.0028238941391735195, 0.0001172907345234568, 0.0021674562153251075, 0.0019494140785282505, 0.018351059142160658, 0.06395343941232096, 0.0018987773569777652, 0.0015121333601829882, 0.002625980651088924], [0.005298412562004844, 0.0897155703508639, 0.0017919867514387797, 0.002886310337255209, 0.0004779776016791037, 0.11759796814627618, 0.005820069939190747, 0.07256559484956768, 0.12023387817391482, 0.0007660342135821575, 0.27194382597257827, 0.0027314877688464695, 0.033094422207379424, 0.0031896781716398218, 0.0667474716208395, 0.2246547619276308, 0.09066098304007646, 0.0008935892149955994, 0.0020287472968355843, 0.06909612413186476, 0.001696602996368597, 0.0915696594568858, 0.2703407740792417, 0.20741230894401746, 0.09133847585683147, 0.03596700462955666, 0.001545622105040766, 0.0010728520806432829, 0.0024862061295124697, 0.007062206619425476, 0.0030016841423244757, 0.01749438643300002, 0.0043309468109449, 0.0029539931653073895, 0.00012276314433319556, 0.0022728459181932713, 0.0020224510885158537, 0.019301856142344778, 0.06709774352664911, 0.002000721294398882, 0.0015697966504950626, 0.0027803266091007027], [0.005660443898706177, 0.09584497195735778, 0.0019288829600131387, 0.0030628793556946262, 0.0005081261474877216, 0.12581894136201427, 0.006202973788607537, 0.07759489669251045, 0.12860052464155589, 0.0008234033705432289, 0.28987273786596895, 0.002890475688942586, 0.03564736735508137, 0.0034165417163123444, 0.07017544736532405, 0.23916444766015943, 0.09621095882458724, 0.0009524884852478748, 0.002178926535596599, 0.07314694725595007, 0.0018083750780301523, 0.09811451360964035, 0.29037693706187123, 0.2200116737840186, 0.0967135234735295, 0.03830001394187229, 0.0016662160468374362, 0.0011535700473194956, 0.002605536867323052, 0.007430090003132482, 0.003206369053881787, 0.01852934214971113, 0.004561093324787513, 0.0031610792666795355, 0.00013124879749694927, 0.002417129446559634, 0.002161500745377406, 0.020319295439106173, 0.0716868642287969, 0.0021141854928115006, 0.0016903021041823538, 0.002913600430642982], [0.006136951561172232, 0.10319131541165144, 0.0020642352949547917, 0.003285756773288506, 0.0005477257453171399, 0.13453520734561897, 0.006695335554010717, 0.08314543128038832, 0.13737715951092533, 0.000886441565301767, 0.310698857497407, 0.0031328555425410055, 0.03790383035693966, 0.003627932139083149, 0.07617331228867123, 0.2605023145387998, 0.1023477931930923, 0.0010052490010318581, 0.00236778632605334, 0.07832357062465223, 0.0019290330732321173, 0.10387577754194334, 0.3113687483556915, 0.23777196982985827, 0.10269039740280583, 0.040868625190884536, 0.0017614270626901208, 0.001237719173110994, 0.0028876009465522703, 0.00791880291273378, 0.003424024577483294, 0.0201782305282521, 0.0049046495343502685, 0.0034254820053703497, 0.00014012020249097262, 0.002606349452005124, 0.0023274870403055437, 0.022307275245406846, 0.07824120171741819, 0.0022295034992477207, 0.001810571469505506, 0.003202114714706445], [0.006550318742212279, 0.11185820950184777, 0.0022662627199609656, 0.003609656463891744, 0.0005975550153341746, 0.14734522996721458, 0.007179400885641709, 0.09231539313029126, 0.14965973597906782, 0.0009568240486010296, 0.3350600432102282, 0.0033534995808502084, 0.041849945580721956, 0.003914858492998698, 0.0814831899045689, 0.28303591856658117, 0.11174426194498431, 0.0011029483672726883, 0.0025609561740630713, 0.0861238166772897, 0.002108678536610318, 0.11288621200093979, 0.34663050739393714, 0.257458943569742, 0.11393998753324344, 0.044954507273257595, 0.001943501262397224, 0.0013556725395262894, 0.003125297773549856, 0.008698174077937115, 0.003676054869836387, 0.022017861931039433, 0.005492302848013321, 0.0036442974211552816, 0.00015258988477229656, 0.0028180500527164804, 0.002553816771224832, 0.02395217812323089, 0.08438830400880239, 0.0024788212487921886, 0.0019813514480644585, 0.003457673308848641], [0.00727087412245189, 0.12225482585344026, 0.0024605247522362715, 0.00397276593188378, 0.000648937053801262, 0.16110531501466627, 0.008029998372237342, 0.10008672436090878, 0.16485838317001622, 0.0010405488380445579, 0.37491202001368695, 0.0037373029494984227, 0.04581796432331777, 0.004365601708103583, 0.08985585859985276, 0.30502977976283996, 0.12427445548241554, 0.0012061475196472027, 0.0028025231206869022, 0.09310364088687947, 0.002301624899399336, 0.1227809104103368, 0.3710228147149438, 0.2822130472849168, 0.12338623403024437, 0.048285172373955125, 0.002161618452057875, 0.001464981869964434, 0.0033737730882617395, 0.00962403426446639, 0.004085918809185392, 0.02425331128766696, 0.005983653431851234, 0.0040264529262639456, 0.00016562487823483957, 0.003175109922941253, 0.0027675212328914906, 0.02610039483966226, 0.013179737681927832, 0.0054824188742823785, 0.031345344820719906, 0.007846136487728408, 0.00539336718655661, 0.0002298899023369318, 0.004155331201201411, 0.0037240950431526435, 0.0352159824740591, 0.12275321063678024, 0.0036067851666619507, 0.002906837541336988, 0.005109534532895032]]
fv_1 = [[0.004853527567298211, 0.08270415337840736, 0.0016554065197528043, 0.002652038605432648, 0.00043725012115650814, 0.1071959879406969, 0.005327365641279487, 0.06714210064453999, 0.11098500775757877, 0.0007084390677100904, 0.24979088908890887, 0.0024914923046286206, 0.030650936915913817, 0.0029335691792253915, 0.060928568926459425, 0.2041387738773878, 0.08308714952249051, 0.0008138942024771402, 0.0018738615477551436, 0.06331127275721861, 0.00156194988239834, 0.0837096374107845, 0.25002100210020994, 0.19033654365436545, 0.08287274857866926, 0.03289899375431595, 0.0014246393199960207, 0.0009865695669613857, 0.0022659471018843386, 0.006436182233108007, 0.0027341387418320508, 0.016092143658810325, 0.003955714291909548, 0.0027123451626249245, 0.00011267367021740775, 0.002092257672069732, 0.0018674436864090987, 0.01764118349334933, 0.06160582315510008, 0.0018218753667138, 0.001445962101671745, 0.0025359850845342647], [0.004894138766217711, 0.08348627005640852, 0.0016697643234361698, 0.002671474954514151, 0.00044187210878445096, 0.10801516289302224, 0.00538903306267693, 0.06781503080839119, 0.11200629022241518, 0.0007168553254916849, 0.25218588946575043, 0.0025137241715463845, 0.030900731427854586, 0.00295936248002833, 0.06164389815837776, 0.2059807861578539, 0.08409037330615662, 0.0008256256383130057, 0.0019023909263014933, 0.06384732070875687, 0.0015841796058652143, 0.08467157750134731, 0.2535166382838031, 0.19210756612643418, 0.08358249629265056, 0.0332213951833465, 0.0014397970951054533, 0.0009962198784309905, 0.0022833229451717776, 0.006488237755134504, 0.0027635592119190132, 0.016246584637311583, 0.0040022849341409435, 0.0027444649727536865, 0.00011376681177545398, 0.0021128588301181214, 0.001888631948555626, 0.01778324583761707, 0.062210802202049255, 0.0018416948739576125, 0.0014519757518074792, 0.0025700376667509768], [0.005043986199230888, 0.08571993271765715, 0.0017226966366072902, 0.0027719653633261226, 0.0004558314780519755, 0.11157736838460477, 0.005504656360814008, 0.07011694240505972, 0.11555826620093672, 0.0007330799422788359, 0.25817645732786576, 0.0025959357039422593, 0.03168425447772218, 0.0030607139512554274, 0.06363197206885646, 0.2139394999669867, 0.0863152392151513, 0.0008436745555454873, 0.001955656386819755, 0.06564432089583774, 0.0016334493108839812, 0.08690726667612438, 0.2614426861104372, 0.1970029940460095, 0.08637599943841794, 0.034222494042727224, 0.0014821006747462015, 0.0010266097823545756, 0.0023693196960222727, 0.006673193763948344, 0.0028452520396721167, 0.01678678276439021, 0.004124091855131796, 0.0028238941391735195, 0.0001172907345234568, 0.0021674562153251075, 0.0019494140785282505, 0.018351059142160658, 0.06395343941232096, 0.0018987773569777652, 0.0015121333601829882, 0.002625980651088924], [0.005298412562004844, 0.0897155703508639, 0.0017919867514387797, 0.002886310337255209, 0.0004779776016791037, 0.11759796814627618, 0.005820069939190747, 0.07256559484956768, 0.12023387817391482, 0.0007660342135821575, 0.27194382597257827, 0.0027314877688464695, 0.033094422207379424, 0.0031896781716398218, 0.0667474716208395, 0.2246547619276308, 0.09066098304007646, 0.0008935892149955994, 0.0020287472968355843, 0.06909612413186476, 0.001696602996368597, 0.0915696594568858, 0.2703407740792417, 0.20741230894401746, 0.09133847585683147, 0.03596700462955666, 0.001545622105040766, 0.0010728520806432829, 0.0024862061295124697, 0.007062206619425476, 0.0030016841423244757, 0.01749438643300002, 0.0043309468109449, 0.0029539931653073895, 0.00012276314433319556, 0.0022728459181932713, 0.0020224510885158537, 0.019301856142344778, 0.06709774352664911, 0.002000721294398882, 0.0015697966504950626, 0.0027803266091007027], [0.005660443898706177, 0.09584497195735778, 0.0019288829600131387, 0.0030628793556946262, 0.0005081261474877216, 0.12581894136201427, 0.006202973788607537, 0.07759489669251045, 0.12860052464155589, 0.0008234033705432289, 0.28987273786596895, 0.002890475688942586, 0.03564736735508137, 0.0034165417163123444, 0.07017544736532405, 0.23916444766015943, 0.09621095882458724, 0.0009524884852478748, 0.002178926535596599, 0.07314694725595007, 0.0018083750780301523, 0.09811451360964035, 0.29037693706187123, 0.2200116737840186, 0.0967135234735295, 0.03830001394187229, 0.0016662160468374362, 0.0011535700473194956, 0.002605536867323052, 0.007430090003132482, 0.003206369053881787, 0.01852934214971113, 0.004561093324787513, 0.0031610792666795355, 0.00013124879749694927, 0.002417129446559634, 0.002161500745377406, 0.020319295439106173, 0.0716868642287969, 0.0021141854928115006, 0.0016903021041823538, 0.002913600430642982], [0.006136951561172232, 0.10319131541165144, 0.0020642352949547917, 0.003285756773288506, 0.0005477257453171399, 0.13453520734561897, 0.006695335554010717, 0.08314543128038832, 0.13737715951092533, 0.000886441565301767, 0.310698857497407, 0.0031328555425410055, 0.03790383035693966, 0.003627932139083149, 0.07617331228867123, 0.2605023145387998, 0.1023477931930923, 0.0010052490010318581, 0.00236778632605334, 0.07832357062465223, 0.0019290330732321173, 0.10387577754194334, 0.3113687483556915, 0.23777196982985827, 0.10269039740280583, 0.040868625190884536, 0.0017614270626901208, 0.001237719173110994, 0.0028876009465522703, 0.00791880291273378, 0.003424024577483294, 0.0201782305282521, 0.0049046495343502685, 0.0034254820053703497, 0.00014012020249097262, 0.002606349452005124, 0.0023274870403055437, 0.022307275245406846, 0.07824120171741819, 0.0022295034992477207, 0.001810571469505506, 0.003202114714706445], [0.006550318742212279, 0.11185820950184777, 0.0022662627199609656, 0.003609656463891744, 0.0005975550153341746, 0.14734522996721458, 0.007179400885641709, 0.09231539313029126, 0.14965973597906782, 0.0009568240486010296, 0.3350600432102282, 0.0033534995808502084, 0.041849945580721956, 0.003914858492998698, 0.0814831899045689, 0.28303591856658117, 0.11174426194498431, 0.0011029483672726883, 0.0025609561740630713, 0.0861238166772897, 0.002108678536610318, 0.11288621200093979, 0.34663050739393714, 0.257458943569742, 0.11393998753324344, 0.044954507273257595, 0.001943501262397224, 0.0013556725395262894, 0.003125297773549856, 0.008698174077937115, 0.003676054869836387, 0.022017861931039433, 0.005492302848013321, 0.0036442974211552816, 0.00015258988477229656, 0.0028180500527164804, 0.002553816771224832, 0.02395217812323089, 0.08438830400880239, 0.0024788212487921886, 0.0019813514480644585, 0.003457673308848641], [0.00727087412245189, 0.12225482585344026, 0.0024605247522362715, 0.00397276593188378, 0.000648937053801262, 0.16110531501466627, 0.008029998372237342, 0.10008672436090878, 0.16485838317001622, 0.0010405488380445579, 0.37491202001368695, 0.0037373029494984227, 0.04581796432331777, 0.004365601708103583, 0.08985585859985276, 0.30502977976283996, 0.12427445548241554, 0.0012061475196472027, 0.0028025231206869022, 0.09310364088687947, 0.002301624899399336, 0.1227809104103368, 0.3710228147149438, 0.2822130472849168, 0.12338623403024437, 0.048285172373955125, 0.002161618452057875, 0.001464981869964434, 0.0033737730882617395, 0.00962403426446639, 0.004085918809185392, 0.02425331128766696, 0.005983653431851234, 0.0040264529262639456, 0.00016562487823483957, 0.003175109922941253, 0.0027675212328914906, 0.026100377461553723, 0.09201891307732166, 0.002734280158455428, 0.0021493526195329273, 0.0037107939583180104], [0.007980944813436102, 0.13628663062528212, 0.002771464390545475, 0.004256123146837298, 0.000718948340175865, 0.17520656515524036, 0.008787202715079672, 0.11120831602929476, 0.18447443196389807, 0.001182941657522351, 0.4059303319961815, 0.0040528368826539905, 0.051017375276520664, 0.0049120970511605236, 0.09834643257993293, 0.3361752436181971, 0.13368255660271627, 0.001324720307264201, 0.003073088156610852, 0.10410670297760632, 0.0025854653226856, 0.13647918228296335, 0.40962485737459203, 0.3190484669598175, 0.1355867557432197, 0.053957229911597016, 0.002324736049025984, 0.0016507960215562084, 0.0037745822822238912, 0.010443350336733121, 0.0044966116832460224, 0.026341878109148373, 0.006450798359423007, 0.004421418493572167, 0.00018399814218718935, 0.003379987488997892, 0.003055018194441801, 0.02896279445950979, 0.10178444522363829, 0.0030293283040150067, 0.0023815813254447343, 0.004169097950555295], [0.00874992945932417, 0.15161755062341353, 0.003037048636280199, 0.004772546666371788, 0.0008039640601834191, 0.19206631788101647, 0.009645603995617757, 0.12130161313245182, 0.20144467974399852, 0.0013025067493723996, 0.4534695872354592, 0.004507268233706943, 0.05633783320107855, 0.005315126415237064, 0.1108314261666619507, 0.002906837541336988, 0.005109534532895032]]

plt.plot(dampenngFactors, acc_0, label='RandomForestClassifier')
plt.plot(dampenngFactors, acc_1, label='SVC')
plt.legend()
plt.show()

#plt.plot(feature_names, average_parameter_value_0[0], label='RandomForestClassifier')
#plt.plot(feature_names, average_parameter_value_1[0], label='SVC')
#plt.legend()
#plt.show()


df_test = pd.DataFrame(apv_0, columns=feature_names)
print(df_test.head())

df = pd.DataFrame(dict(
    x = apv_0[0],
    y = feature_names
))

fig = px.line(df, x="x", y="y", title="LALALA", color='y') 
fig.show()'''