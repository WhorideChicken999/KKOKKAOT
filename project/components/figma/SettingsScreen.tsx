import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Switch, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronLeft } from 'lucide-react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

type Settings = { darkMode: boolean; beta: boolean; lang: 'ko'|'en' };
type Props = { onBack?: () => void };

export default function SettingsScreen({ onBack }: Props) {
  const [s, setS] = useState<Settings>({ darkMode:false, beta:false, lang:'ko' });

  useEffect(() => { (async () => {
    const raw = await AsyncStorage.getItem('@kko/settings');
    if (raw) setS(JSON.parse(raw));
  })(); }, []);

  const save = async (next: Partial<Settings>) => {
    const merged = { ...s, ...next }; setS(merged);
    await AsyncStorage.setItem('@kko/settings', JSON.stringify(merged));
  };

  return (
    <View style={SS.wrap}>
      {onBack && (
        <SafeAreaView edges={['top']} style={{ backgroundColor:'#fff' }}>
          <View style={H.bar}>
            <Pressable onPress={onBack} hitSlop={12} style={H.back}>
              <ChevronLeft size={24} color="#111" />
            </Pressable>
            <Text style={H.title}>설정</Text>
            <View style={{ width: 36 }} />
          </View>
        </SafeAreaView>
      )}

      <View style={SS.card}>
        <Row label="다크 모드">
          <Switch value={s.darkMode} onValueChange={(v)=>save({darkMode:v})}/>
        </Row>
        <Row label="분석 베타 기능">
          <Switch value={s.beta} onValueChange={(v)=>save({beta:v})}/>
        </Row>
        <Row label="언어">
          <View style={{flexDirection:'row',gap:8}}>
            <Chip selected={s.lang==='ko'} onPress={()=>save({lang:'ko'})} label="한국어"/>
            <Chip selected={s.lang==='en'} onPress={()=>save({lang:'en'})} label="English"/>
          </View>
        </Row>
      </View>

      <Text style={SS.tip}>설정은 이 기기에서만 저장돼요.</Text>
    </View>
  );
}

function Row({ label, children }: any){
  return <View style={SS.row}><Text style={SS.label}>{label}</Text><View>{children}</View></View>;
}
function Chip({ selected, label, onPress }: any){
  return <Pressable onPress={onPress} style={[SS.chip, selected&&SS.chipOn]}>
    <Text style={[SS.chipTx, selected&&SS.chipTxOn]}>{label}</Text>
  </Pressable>;
}

const H = StyleSheet.create({
  bar:{height:56,paddingHorizontal:12,flexDirection:'row',alignItems:'center',
       borderBottomWidth:StyleSheet.hairlineWidth,borderBottomColor:'#eee'},
  back:{padding:6},
  title:{flex:1,textAlign:'center',fontSize:18,fontWeight:'700',color:'#111'},
});

const SS = StyleSheet.create({
  wrap:{flex:1,padding:20,backgroundColor:'#fafafa',gap:16},
  card:{backgroundColor:'#fff',borderRadius:14,padding:12,gap:8,elevation:1,
        shadowColor:'#000',shadowOpacity:0.05,shadowRadius:6},
  row:{flexDirection:'row',justifyContent:'space-between',alignItems:'center',
       paddingVertical:12,borderBottomColor:'#eee',borderBottomWidth:1},
  label:{fontSize:16},
  chip:{paddingHorizontal:12,paddingVertical:8,borderRadius:999,backgroundColor:'#f1f1f1'},
  chipOn:{backgroundColor:'#191919'},
  chipTx:{color:'#222',fontWeight:'600'},
  chipTxOn:{color:'#fff'},
  tip:{color:'#777',fontSize:12},
});
