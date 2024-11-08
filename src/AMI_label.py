import os,glob
import numpy as np
# pip install pyannote.audio
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import itertools


meeting_info = {'EN2001a': {'0': 'MEE068', '1': 'FEO066', '2': 'FEO065', '3': 'MEE067', '4': 'MEO069'}, 'EN2001b': {'0': 'FEO066', '1': 'MEO069', '2': 'FEO065', '3': 'MEE068'}, 'EN2001d': {'0': 'FEO066', '1': 'MEO069', '2': 'FEO065', '3': 'MEE067', '4': 'MEE068'}, 'EN2001e': {'0': 'FEO066', '1': 'MEO069', '2': 'FEO065', '3': 'MEE067', '4': 'MEE068'}, 'EN2002a': {'0': 'MEE073', '1': 'FEO070', '2': 'FEO072', '3': 'MEE071'}, 'EN2002b': {'0': 'FEO070', '1': 'MEE071', '2': 'FEO072', '3': 'MEE073'}, 'EN2002c': {'1': 'FEO072', '2': 'MEE071', '3': 'MEE073'}, 'EN2002d': {'0': 'FEO070', '1': 'FEO072', '2': 'MEE071', '3': 'MEE073'}, 'EN2003a': {'0': 'MEE075', '1': 'MEE076', '3': 'MEO074'}, 'EN2004a': {'0': 'FEE080', '1': 'FEE078', '2': 'FEO079', '3': 'FEE081'}, 'EN2005a': {'0': 'MEO082', '1': 'FEE083', '2': 'FEO084', '3': 'FEE085'}, 'EN2006a': {'0': 'MEE089', '1': 'FEE087', '2': 'FEE088', '3': 'MEO086'}, 'EN2006b': {'0': 'MEE089', '1': 'FEE087', '2': 'FEE088', '3': 'MEO086'}, 'EN2009b': {'0': 'MEE095', '1': 'FEE083', '2': 'MEE094'}, 'EN2009c': {'0': 'MEE095', '1': 'FEE083', '2': 'MEE094'}, 'EN2009d': {'0': 'FEE083', '1': 'FEE096', '2': 'MEE094', '3': 'MEE095'}, 'ES2002a': {'0': 'MEE006', '1': 'FEE005', '2': 'MEE007', '3': 'MEE008'}, 'ES2002b': {'0': 'MEE006', '1': 'FEE005', '2': 'MEE007', '3': 'MEE008'}, 'ES2002c': {'0': 'MEE006', '1': 'FEE005', '2': 'MEE007', '3': 'MEE008'}, 'ES2002d': {'0': 'MEE006', '1': 'FEE005', '2': 'MEE008', '3': 'MEE007'}, 'ES2003a': {'0': 'MEE011', '1': 'MEE009', '2': 'MEE012', '3': 'MEE010'}, 'ES2003b': {'0': 'MEE011', '1': 'MEE009', '2': 'MEE012', '3': 'MEE010'}, 'ES2003c': {'0': 'MEE011', '1': 'MEE009', '2': 'MEE012', '3': 'MEE010'}, 'ES2003d': {'0': 'MEE011', '1': 'MEE009', '2': 'MEE012', '3': 'MEE010'}, 'ES2004a': {'0': 'MEO015', '1': 'FEE013', '2': 'MEE014', '3': 'FEE016'}, 'ES2004b': {'0': 'MEO015', '1': 'FEE013', '2': 'MEE014', '3': 'FEE016'}, 'ES2004c': {'0': 'MEO015', '1': 'FEE013', '2': 'MEE014', '3': 'FEE016'}, 'ES2004d': {'0': 'MEO015', '1': 'FEE013', '2': 'MEE014', '3': 'FEE016'}, 'ES2005a': {'0': 'MEE018', '1': 'MEO020', '2': 'MEE017', '3': 'FEE019'}, 'ES2005b': {'0': 'MEE018', '1': 'MEO020', '2': 'MEE017', '3': 'FEE019'}, 'ES2005c': {'0': 'MEE018', '1': 'MEO020', '2': 'MEE017', '3': 'FEE019'}, 'ES2005d': {'0': 'MEE018', '1': 'MEO020', '2': 'MEE017', '3': 'FEE019'}, 'ES2006a': {'0': 'MEO022', '1': 'FEE024', '2': 'FEE021', '3': 'FEO023'}, 'ES2006b': {'0': 'MEO022', '1': 'FEE024', '2': 'FEE021', '3': 'FEO023'}, 'ES2006c': {'0': 'MEO022', '1': 'FEE024', '2': 'FEE021', '3': 'FEO023'}, 'ES2006d': {'0': 'MEO022', '1': 'FEE024', '2': 'FEE021', '3': 'FEO023'}, 'ES2007a': {'0': 'MEE025', '1': 'FEO026', '2': 'MEE027', '3': 'FEE028'}, 'ES2007b': {'0': 'MEE025', '1': 'FEO026', '2': 'MEE027', '3': 'FEE028'}, 'ES2007c': {'0': 'MEE025', '1': 'FEO026', '2': 'MEE027', '3': 'FEE028'}, 'ES2007d': {'0': 'MEE025', '1': 'FEO026', '2': 'MEE027', '3': 'FEE028'}, 'ES2008a': {'0': 'FEE029', '1': 'FEE030', '2': 'MEE031', '3': 'FEE032'}, 'ES2008b': {'0': 'FEE029', '1': 'FEE030', '2': 'MEE031', '3': 'FEE032'}, 'ES2008c': {'0': 'FEE029', '1': 'FEE030', '2': 'MEE031', '3': 'FEE032'}, 'ES2008d': {'0': 'FEE029', '1': 'FEE030', '2': 'MEE031', '3': 'FEE032'}, 'ES2009a': {'0': 'MEE033', '1': 'MEE034', '2': 'MEE035', '3': 'FEE036'}, 'ES2009b': {'0': 'MEE033', '1': 'MEE034', '2': 'MEE035', '3': 'FEE036'}, 'ES2009c': {'0': 'MEE033', '1': 'MEE034', '2': 'MEE035', '3': 'FEE036'}, 'ES2009d': {'0': 'MEE033', '1': 'MEE034', '2': 'MEE035', '3': 'FEE036'}, 'ES2010a': {'0': 'FEE037', '1': 'FEE038', '2': 'FEE039', '3': 'FEE040'}, 'ES2010b': {'0': 'FEE037', '1': 'FEE038', '2': 'FEE039', '3': 'FEE040'}, 'ES2010c': {'0': 'FEE037', '1': 'FEE038', '2': 'FEE039', '3': 'FEE040'}, 'ES2010d': {'0': 'FEE037', '1': 'FEE038', '2': 'FEE039', '3': 'FEE040'}, 'ES2011a': {'0': 'FEE041', '1': 'FEE042', '2': 'FEE043', '3': 'FEE044'}, 'ES2011b': {'0': 'FEE041', '1': 'FEE042', '2': 'FEE043', '3': 'FEE044'}, 'ES2011c': {'0': 'FEE041', '1': 'FEE042', '2': 'FEE043', '3': 'FEE044'}, 'ES2011d': {'0': 'FEE041', '1': 'FEE042', '2': 'FEE043', '3': 'FEE044'}, 'ES2012a': {'0': 'MEE045', '1': 'FEE046', '2': 'FEE047', '3': 'MEE048'}, 'ES2012b': {'0': 'MEE045', '1': 'FEE046', '2': 'FEE047', '3': 'MEE048'}, 'ES2012c': {'0': 'MEE045', '1': 'FEE046', '2': 'FEE047', '3': 'MEE048'}, 'ES2012d': {'0': 'MEE045', '1': 'FEE046', '2': 'FEE047', '3': 'MEE048'}, 'ES2013a': {'0': 'FEE049', '1': 'FEE050', '2': 'FEE051', '3': 'FEE052'}, 'ES2013b': {'0': 'FEE049', '1': 'FEE050', '2': 'FEE051', '3': 'FEE052'}, 'ES2013c': {'0': 'FEE049', '1': 'FEE050', '2': 'FEE051', '3': 'FEE052'}, 'ES2013d': {'0': 'FEE049', '1': 'FEE050', '2': 'FEE051', '3': 'FEE052'}, 'ES2014a': {'0': 'MEE053', '1': 'MEE054', '2': 'FEE055', '3': 'MEE056'}, 'ES2014b': {'0': 'MEE053', '1': 'MEE054', '2': 'FEE055', '3': 'MEE056'}, 'ES2014c': {'0': 'MEE053', '1': 'MEE054', '2': 'FEE055', '3': 'MEE056'}, 'ES2014d': {'0': 'MEE053', '1': 'MEE054', '2': 'FEE055', '3': 'MEE056'}, 'ES2015a': {'0': 'FEE057', '1': 'FEE058', '2': 'FEE059', '3': 'FEE060'}, 'ES2015b': {'0': 'FEE057', '1': 'FEE058', '2': 'FEE059', '3': 'FEE060'}, 'ES2015c': {'0': 'FEE057', '1': 'FEE058', '2': 'FEE059', '3': 'FEE060'}, 'ES2015d': {'0': 'FEE057', '1': 'FEE058', '2': 'FEE059', '3': 'FEE060'}, 'ES2016a': {'0': 'MEE061', '1': 'MEO062', '2': 'MEE063', '3': 'FEE064'}, 'ES2016b': {'0': 'MEE061', '1': 'MEO062', '2': 'MEE063', '3': 'FEE064'}, 'ES2016c': {'0': 'MEE061', '1': 'MEO062', '2': 'MEE063', '3': 'FEE064'}, 'ES2016d': {'0': 'MEE061', '1': 'MEO062', '2': 'MEE063', '3': 'FEE064'}, 'IB4001': {'0': 'MIO092', '1': 'FIE038', '2': 'FIO093', '3': 'MIO091'}, 'IB4002': {'0': 'MIO092', '1': 'FIE038', '2': 'FIO093', '3': 'MIO091'}, 'IB4003': {'0': 'MIO036', '1': 'MIO094', '2': 'FIE037', '3': 'MIO039'}, 'IB4004': {'0': 'MIO036', '1': 'MIO094', '2': 'FIE037', '3': 'MIO039'}, 'IB4005': {'0': 'FIE038', '1': 'MIE032', '2': 'MIO036', '3': 'MIO078'}, 'IB4010': {'0': 'FIE038', '1': 'MIO036', '2': 'MIO095', '3': 'MIO046'}, 'IB4011': {'0': 'FIE038', '1': 'MIO036', '2': 'MIO095', '3': 'MIO046'}, 'IN1001': {'0': 'MIO024', '1': 'MIO016', '2': 'MIO066'}, 'IN1002': {'0': 'MIO022', '1': 'MIO076', '2': 'MIO050', '3': 'MIO078'}, 'IN1005': {'0': 'FIO041', '1': 'MIE080', '2': 'MIO043', '3': 'MIO023'}, 'IN1007': {'0': 'MIO098', '1': 'MIO097', '2': 'MIO106', '3': 'MIO077'}, 'IN1008': {'0': 'MIO100', '1': 'MIO099', '2': 'MIO101', '3': 'MIO018'}, 'IN1009': {'0': 'MIO100', '1': 'MIO099', '2': 'MIO101', '3': 'MIO018'}, 'IN1012': {'0': 'MIO078', '1': 'MIO097', '2': 'MIO077', '3': 'MIO018'}, 'IN1013': {'0': 'MIE034', '1': 'MIO078', '2': 'MIO105', '3': 'MIO097'}, 'IN1014': {'0': 'MIO024', '1': 'MIO020', '2': 'MIO016', '3': 'MIO031'}, 'IN1016': {'0': 'MIO104', '1': 'MIO031', '2': 'MIE034', '3': 'MIO050'}, 'IS1000a': {'0': 'FIE081', '1': 'MIO082', '2': 'MIO050', '3': 'MIO016'}, 'IS1000b': {'0': 'FIE081', '1': 'MIO082', '2': 'MIO050', '3': 'MIO016'}, 'IS1000c': {'0': 'FIE081', '1': 'MIO082', '2': 'MIO050', '3': 'MIO016'}, 'IS1000d': {'0': 'FIE081', '1': 'MIO082', '2': 'MIO016', '3': 'MIO050'}, 'IS1001a': {'0': 'MIO043', '1': 'MIO012', '2': 'MIO020', '3': 'FIO074'}, 'IS1001b': {'0': 'MIO043', '1': 'MIO012', '2': 'MIO020', '3': 'FIO074'}, 'IS1001c': {'0': 'MIO043', '1': 'MIO020', '2': 'MIO012', '3': 'FIO074'}, 'IS1001d': {'0': 'MIO043', '1': 'MIO020', '2': 'MIO012', '3': 'FIO074'}, 'IS1002b': {'0': 'MIE080', '1': 'MIE029', '2': 'MIO026', '3': 'MIE083'}, 'IS1002c': {'0': 'MIE080', '1': 'MIE029', '2': 'MIO026', '3': 'MIE083'}, 'IS1002d': {'0': 'MIE080', '1': 'MIE029', '2': 'MIO026', '3': 'MIE083'}, 'IS1003a': {'0': 'FIO017', '1': 'MIO035', '2': 'MIO005', '3': 'MIO023'}, 'IS1003b': {'0': 'MIO023', '1': 'MIO005', '2': 'FIO017', '3': 'MIO035'}, 'IS1003c': {'0': 'FIO017', '1': 'MIO005', '2': 'MIO035', '3': 'MIO023'}, 'IS1003d': {'0': 'FIO017', '1': 'MIO005', '2': 'MIO035', '3': 'MIO023'}, 'IS1004a': {'0': 'MIO019', '1': 'MIE090', '2': 'MIO022', '3': 'MIO047'}, 'IS1004b': {'0': 'MIO019', '1': 'MIE090', '2': 'MIO022', '3': 'MIO047'}, 'IS1004c': {'0': 'MIO019', '1': 'MIE090', '2': 'MIO022', '3': 'MIO047'}, 'IS1004d': {'0': 'MIO019', '1': 'MIE090', '2': 'MIO022', '3': 'MIO047'}, 'IS1005a': {'0': 'MIO055', '1': 'MIO077', '2': 'MIO076', '3': 'MIE002'}, 'IS1005b': {'0': 'MIO055', '1': 'MIO077', '2': 'MIO076', '3': 'MIE002'}, 'IS1005c': {'0': 'MIO055', '1': 'MIO077', '2': 'MIO076', '3': 'MIE002'}, 'IS1006a': {'0': 'FIO041', '1': 'MIO040', '2': 'MIO078', '3': 'MIO008'}, 'IS1006b': {'0': 'FIO041', '1': 'MIO040', '2': 'MIO078', '3': 'MIO008'}, 'IS1006c': {'0': 'FIO041', '1': 'MIO040', '2': 'MIO078', '3': 'MIO008'}, 'IS1006d': {'0': 'FIO041', '1': 'MIO040', '2': 'MIO078', '3': 'MIO008'}, 'IS1007a': {'0': 'MIO049', '1': 'MIO025', '2': 'MIO075', '3': 'MIO072'}, 'IS1007b': {'0': 'MIO049', '1': 'MIO025', '2': 'MIO075', '3': 'MIO072'}, 'IS1007c': {'0': 'MIO049', '1': 'MIO025', '2': 'MIO075', '3': 'MIO072'}, 'IS1007d': {'0': 'MIO049', '1': 'MIO025', '2': 'MIO075', '3': 'MIO072'}, 'IS1008a': {'0': 'MIO086', '1': 'FIE038', '2': 'FIE073', '3': 'MIE085'}, 'IS1008b': {'0': 'MIO086', '1': 'FIE073', '2': 'FIE038', '3': 'MIE085'}, 'IS1008c': {'0': 'MIO086', '1': 'FIE038', '2': 'FIE073', '3': 'MIE085'}, 'IS1008d': {'0': 'MIO086', '1': 'FIE073', '2': 'FIE038', '3': 'MIE085'}, 'IS1009a': {'0': 'FIE088', '1': 'FIO087', '2': 'FIO084', '3': 'FIO089'}, 'IS1009b': {'0': 'FIE088', '1': 'FIO087', '2': 'FIO084', '3': 'FIO089'}, 'IS1009c': {'0': 'FIE088', '1': 'FIO087', '2': 'FIO084', '3': 'FIO089'}, 'IS1009d': {'0': 'FIE088', '1': 'FIO087', '2': 'FIO084', '3': 'FIO089'}, 'TS3003a': {'0': 'MTD009PM', '1': 'MTD011UID', '2': 'MTD0010ID', '3': 'MTD012ME'}, 'TS3003b': {'0': 'MTD009PM', '1': 'MTD011UID', '2': 'MTD0010ID', '3': 'MTD012ME'}, 'TS3003c': {'0': 'MTD009PM', '1': 'MTD011UID', '2': 'MTD0010ID', '3': 'MTD012ME'}, 'TS3003d': {'0': 'MTD009PM', '1': 'MTD011UID', '2': 'MTD0010ID', '3': 'MTD012ME'}, 'TS3004a': {'0': 'MTD013PM', '1': 'MTD015UID', '2': 'MTD014ID', '3': 'MTD016ME'}, 'TS3004b': {'0': 'MTD013PM', '1': 'MTD015UID', '2': 'MTD014ID', '3': 'MTD016ME'}, 'TS3004c': {'0': 'MTD013PM', '1': 'MTD015UID', '2': 'MTD014ID', '3': 'MTD016ME'}, 'TS3004d': {'0': 'MTD013PM', '1': 'MTD015UID', '2': 'MTD014ID', '3': 'MTD016ME'}, 'TS3005a': {'0': 'MTD017PM', '1': 'FTD019UID', '2': 'MTD018ID', '3': 'MTD020ME'}, 'TS3005b': {'0': 'MTD017PM', '1': 'FTD019UID', '2': 'MTD018ID', '3': 'MTD020ME'}, 'TS3005c': {'0': 'MTD017PM', '1': 'FTD019UID', '2': 'MTD018ID', '3': 'MTD020ME'}, 'TS3005d': {'0': 'MTD017PM', '1': 'FTD019UID', '2': 'MTD018ID', '3': 'MTD020ME'}, 'TS3006a': {'0': 'MTD021PM', '1': 'MTD023UID', '2': 'MTD022ID', '3': 'MTD024ME'}, 'TS3006b': {'0': 'MTD021PM', '1': 'MTD023UID', '2': 'MTD022ID', '3': 'MTD024ME'}, 'TS3006c': {'0': 'MTD021PM', '1': 'MTD023UID', '2': 'MTD022ID', '3': 'MTD024ME'}, 'TS3006d': {'0': 'MTD021PM', '1': 'MTD023UID', '2': 'MTD022ID', '3': 'MTD024ME'}, 'TS3007a': {'0': 'MTD025PM', '1': 'MTD027ID', '2': 'MTD026UID', '3': 'MTD028ME'}, 'TS3007b': {'0': 'MTD025PM', '1': 'MTD027ID', '2': 'MTD026UID', '3': 'MTD028ME'}, 'TS3007c': {'0': 'MTD025PM', '1': 'MTD027ID', '2': 'MTD026UID', '3': 'MTD028ME'}, 'TS3007d': {'0': 'MTD025PM', '1': 'MTD027ID', '2': 'MTD026UID', '3': 'MTD028ME'}, 'TS3008a': {'0': 'MTD029PM', '1': 'MTD031UID', '2': 'MTD030ID', '3': 'MTD032ME'}, 'TS3008b': {'0': 'MTD029PM', '1': 'MTD031UID', '2': 'MTD030ID', '3': 'MTD032ME'}, 'TS3008c': {'0': 'MTD029PM', '1': 'MTD031UID', '2': 'MTD030ID', '3': 'MTD032ME'}, 'TS3008d': {'0': 'MTD029PM', '1': 'MTD031UID', '2': 'MTD030ID', '3': 'MTD032ME'}, 'TS3009a': {'0': 'MTD033PM', '1': 'MTD035UID', '2': 'MTD034ID', '3': 'MTD036ME'}, 'TS3009b': {'0': 'MTD033PM', '1': 'MTD035UID', '2': 'MTD034ID', '3': 'MTD036ME'}, 'TS3009c': {'0': 'MTD033PM', '1': 'MTD035UID', '2': 'MTD034ID', '3': 'MTD036ME'}, 'TS3009d': {'0': 'MTD033PM', '1': 'MTD035UID', '2': 'MTD034ID', '3': 'MTD036ME'}, 'TS3010a': {'0': 'MTD037PM', '1': 'MTD039UID', '2': 'MTD038ID', '3': 'MTD040ME'}, 'TS3010b': {'0': 'MTD037PM', '1': 'MTD039UID', '2': 'MTD038ID', '3': 'MTD040ME'}, 'TS3010c': {'0': 'MTD037PM', '1': 'MTD039UID', '2': 'MTD038ID', '3': 'MTD040ME'}, 'TS3010d': {'0': 'MTD037PM', '1': 'MTD039UID', '2': 'MTD038ID', '3': 'MTD040ME'}, 'TS3011a': {'0': 'MTD041PM', '1': 'MTD043UID', '2': 'MTD042ID', '3': 'MTD044ME'}, 'TS3011b': {'0': 'MTD041PM', '1': 'MTD043UID', '2': 'MTD042ID', '3': 'MTD044ME'}, 'TS3011c': {'0': 'MTD041PM', '1': 'MTD043UID', '2': 'MTD042ID', '3': 'MTD044ME'}, 'TS3011d': {'0': 'MTD041PM', '1': 'MTD043UID', '2': 'MTD042ID', '3': 'MTD044ME'}, 'TS3012a': {'0': 'MTD045PM', '1': 'MTD047UID', '2': 'MTD046ID', '3': 'MTD048ME'}, 'TS3012b': {'0': 'MTD045PM', '1': 'MTD047UID', '2': 'MTD046ID', '3': 'MTD048ME'}, 'TS3012c': {'0': 'MTD045PM', '1': 'MTD047UID', '2': 'MTD046ID', '3': 'MTD048ME'}, 'TS3012d': {'0': 'MTD045PM', '1': 'MTD047UID', '2': 'MTD046ID', '3': 'MTD048ME'}}


class AMI_label():
    def __init__(self, dir_label, target="only_words"):
        self.data = {}
        # Label example for ES2002a.Closeup.avi
        # path = /home/data/kbh/AMI_IITP/amicorpus/ES2002a/video/ES2002a.Closeup1.avi
        # SpeakerID : meeting_info["ES2002a"]["0"] == "MEE006"
        # Label /../AMI-diarization-setup/<only_words/word_and_vocalsoduns>/rttms/<dev/train/test>/ES2002a.rttm
        #   SPEAKER ES2002a 1 74.89 0.35 <NA> <NA> MEE008 <NA> <NA>
        #   SPEAKER ES2002a 1 77.44 3.43 <NA> <NA> MEE006 <NA> <NA>
        #   -> speaker_id = line[7]
        #   -> start_sec = line[3]
        #   -> end_sec = line[3] + line[4]

        # Label for each meeting
        label_list = glob.glob(os.path.join(dir_label,target,"rttms","**",'*.rttm'))

        print(f"AMI_Label :{dir_label}-> {len(label_list)}")

        # per meeting
        for label in label_list:
            name = label.split('/')[-1]
            meeting_id = name.split('.')[0]
            self.data[meeting_id] = {}
            #print("{} {}".format(meeting_id,self.data[meeting_id]))

            dir_cat = label.split('/')[-2]
            path_uems = os.path.join(dir_label,"uems",dir_cat,meeting_id+".uem")
            with open(path_uems,'r') as f:
                line = f.readline()
                ground_truth_duration = float(line.split()[3])
            self.data[meeting_id]["GTD"] = ground_truth_duration
            self.data[meeting_id]["Label"] = {}

            # load label
            with open(label,'r') as f:
                # temporal key for parsing
                tmp_id = {}
                # per speaker, set up for label
                for meeting in meeting_info[meeting_id]:
                    for key in meeting :
                        speaker_id = meeting_info[meeting_id][key]
                        self.data[meeting_id]["Label"][key] = {"ID":speaker_id,"segs":[]}
                        tmp_id[speaker_id] = key
                #print(self.data[meeting_id])

                # parse label
                lines = f.readlines()
                # per line
                for line in lines:
                    line = line.split()
                    speaker_id = line[7]
                    start_sec = float(line[3])
                    end_sec = start_sec + float(line[4])
                    timestep = (start_sec,end_sec)
                    self.data[meeting_id]["Label"][tmp_id[speaker_id]]["segs"].append(timestep)

    
    def __getitem__(self, key):
        return self.data.get(key, "Key not found")

    """
    + Diarization Error Rate (DER)

    False Alarms (FA): Duration of non-speech segments misclassified as speech.
    Missed Detections (MD): Duration of speech segments misclassified as non-speech.
    Speaker Confusions (SC): Duration of speech segments misattributed to the wrong speaker (misclassified).

    DER = (FA + MD + SC) / Ground Truth Duration

    """
    def measure(self, meeting, estim, unit=0.008, margin=0.25, skip_overlap=False) : 
        ED = estim.shape[1]*unit
        GTD = self.data[meeting]["GTD"]

        if abs(ED - GTD) > 3.0 :
            print("ERROR::Ground Truth Duration is not matched with the estimation {} {}".format(ED, GTD))

        estim_sec = []
        speech_sec = 0

        # estim frame to second
        for idx in range(estim.shape[0]) : 
            cur = 0
            prev = 0
            start = 0
            end = 0
            estim_sec.append([])
            for i in range(estim.shape[1]) : 
                cur = estim[idx][i]
                if cur == prev :
                    continue
                else : 
                    # start of speech
                    if prev == 0:
                        start = i
                    # end of speech
                    else :
                        end  = i
                        start_sec = start * unit
                        end_sec = end * unit
                        dur = end_sec - start_sec
                        if dur < 0.2 :
                            continue

                        speech_sec += dur
                        estim_sec[idx].append((start_sec,end_sec))
                prev = cur

        #print("total speech duration : {}".format(speech_sec))
        ground_truth = Annotation()
        for spk in self.data[meeting]["Label"] : 
            for seg in self.data[meeting]["Label"][spk]["segs"] : 
                ground_truth[Segment(seg[0],seg[1])] = self.data[meeting]["Label"][spk]["ID"]
                #print("GT : ({}, {}) = {}".format(seg[0],seg[1],self.data[meeting]["Label"][spk]["ID"]))
            #print("[{}]".format(self.data[meeting]["Label"][spk]["ID"]))

        best_DER = 100.0
        best_result = None

        # 250 margin on both side
        metric = DiarizationErrorRate(collar=0.5, skip_overlap=skip_overlap)
        hypothesis = Annotation()
        for idx, spk in enumerate(range(4)) : 
            speaker_id = self.data[meeting]["Label"][str(spk)]["ID"]
            #print(speaker_id)
            for seg in estim_sec[idx] : 
                hypothesis[Segment(seg[0],seg[1])] = speaker_id
                #print("HP : ({}, {}) = {}".format(seg[0],seg[1],speaker_id))
            #print("HP{} {}->[{}]".format(perm,idx,speaker_id))

        # calculate DER
        result = metric(ground_truth, hypothesis,detailed=True)
        #print(result["diarization error rate"])
        if result["diarization error rate"] < best_DER :
            best_DER = result["diarization error rate"]
            best_result = result
        return best_DER, best_result


    def measurePIT(self, meeting, estim, unit=0.008, margin=0.25) : 
        pair = []
        ED =estim.shape[1]*unit
        GTD = self.data[meeting]["GTD"]

        if abs(ED - GTD) > 1.0 :
            print("ERROR::Ground Truth Duration is not matched with the estimation {} {}".format(ED, GTD))

        estim_sec = []
        speech_sec = 0

        # estim frame to second
        for idx in range(estim.shape[0]) : 
            cur = 0
            prev = 0
            start = 0
            end = 0
            estim_sec.append([])
            for i in range(estim.shape[1]) : 
                cur = estim[idx][i]
                if cur == prev :
                    continue
                else : 
                    # start of speech
                    if prev == 0:
                        start = i
                    # end of speech
                    else :
                        end  = i
                        start_sec = start * unit
                        end_sec = end * unit
                        dur = end_sec - start_sec
                        if dur < 0.2 :
                            continue

                        speech_sec += dur
                        estim_sec[idx].append((start_sec,end_sec))
                prev = cur

        print("total speech duration : {}".format(speech_sec))
        ground_truth = Annotation()
        for spk in self.data[meeting]["Label"] : 
            for seg in self.data[meeting]["Label"][spk]["segs"] : 
                ground_truth[Segment(seg[0],seg[1])] = self.data[meeting]["Label"][spk]["ID"]
                #print("GT : ({}, {}) = {}".format(seg[0],seg[1],self.data[meeting]["Label"][spk]["ID"]))
            print("[{}]".format(self.data[meeting]["Label"][spk]["ID"]))

        best_DER = 100.0
        best_result = None

        # for every combination of estimated_speaker
        permutations = list(itertools.permutations([0,1,2,3]))
        for perm in permutations : 
            # 250 margin on both side
            metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)
            hypothesis = Annotation()
            for idx, spk in enumerate(perm) : 
                speaker_id = self.data[meeting]["Label"][str(spk)]["ID"]
                #print(speaker_id)
                for seg in estim_sec[idx] : 
                    hypothesis[Segment(seg[0],seg[1])] = speaker_id
                    #print("HP{} : ({}, {}) = {}".format(perm, seg[0],seg[1],speaker_id))
                #print("HP{} {}->[{}]".format(perm,idx,speaker_id))

            # calculate DER
            result = metric(ground_truth, hypothesis,detailed=True)
            #print(result["diarization error rate"])
            if result["diarization error rate"] < best_DER :
                best_DER = result["diarization error rate"]
                best_result = result
        return best_DER, best_result

if __name__ == '__main__':

    import scipy.io
    mat_data = scipy.io.loadmat('ES2002a.Array1-01_v2.mat')
    #mat_data = scipy.io.loadmat('ES2002a.Array1-01.mat')

    label = AMI_label("/home/kbh/work/1_Active/MMVAD/AMI-diarization-setup",target = "word_and_vocalsounds" )
    DER,result = label.measurePIT("ES2002a",mat_data["spk_vad"])
    print("{} {}".format(DER,result))

    label = AMI_label("/home/kbh/work/1_Active/MMVAD/AMI-diarization-setup")
    DER,result = label.measurePIT("ES2002a",mat_data["spk_vad"])
    print("{} {}".format(DER,result))

    """
    reference = Annotation()
    reference[Segment(0, 4.25)] = 'A'
    reference[Segment(3.25,6.25)] = 'B'

    hypothesis = Annotation()
    hypothesis[Segment(0, 4)] = 'A'
    hypothesis[Segment(3.0,6.0)] = 'B'

    metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)
    DER = metric(reference, hypothesis, detailed=True)
    print(DER)

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    DER = metric(reference, hypothesis, detailed=True)
    print(DER)

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    DER = metric(reference, hypothesis, detailed=True)
    print(DER)
    """



    