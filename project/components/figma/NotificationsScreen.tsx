import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronLeft } from 'lucide-react-native';
import * as Notifications from 'expo-notifications';
import * as Linking from 'expo-linking';

type Props = { onBack?: () => void };

export default function NotificationsScreen({ onBack }: Props) {
  const [status, setStatus] = useState<'granted'|'denied'|'undetermined'>('undetermined');
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => { (async () => {
    const s = await Notifications.getPermissionsAsync();
    setStatus(s.status as any);
    if (s.status === 'granted') {
      const t = await Notifications.getExpoPushTokenAsync();
      setToken(t.data);
    }
  })(); }, []);

  const ask = async () => {
    const { status } = await Notifications.requestPermissionsAsync();
    setStatus(status as any);
    if (status !== 'granted') Alert.alert('알림 권한이 거부되었습니다.', '설정에서 직접 허용해 주세요.');
  };

  const openSystem = () => {
    if (Platform.OS === 'ios') Linking.openURL('app-settings:');
    else Linking.openSettings();
  };

  const testLocal = async () => {
    await Notifications.scheduleNotificationAsync({
      content:{ title:'꼬까옷', body:'알림 테스트입니다 👋' }, trigger:null
    });
  };

  return (
    <View style={NS.wrap}>
      {onBack && (
        <SafeAreaView edges={['top']} style={{ backgroundColor:'#fff' }}>
          <View style={H.bar}>
            <Pressable onPress={onBack} hitSlop={12} style={H.back}>
              <ChevronLeft size={24} color="#111" />
            </Pressable>
            <Text style={H.title}>알림</Text>
            <View style={{ width: 36 }} />
          </View>
        </SafeAreaView>
      )}

      <View style={NS.card}>
        <Row label="권한 상태" right={status} />
        <Btn onPress={ask} label="권한 요청/갱신" kind="primary"/>
        <Btn onPress={testLocal} label="로컬 알림 테스트" />
        <Btn onPress={openSystem} label="시스템 설정 열기" />
        {token && <Text selectable style={NS.token}>Expo Push Token: {token}</Text>}
      </View>

      <Text style={NS.tip}>원격 푸시는 서버에서 이 토큰으로 발송해요.</Text>
    </View>
  );
}

function Row({ label, right }: any){
  return <View style={NS.row}><Text style={NS.label}>{label}</Text><Text style={NS.right}>{right}</Text></View>;
}
function Btn({ onPress, label, kind='ghost' }: any){
  return <Pressable onPress={onPress} style={[NS.btn, kind==='primary'?NS.btnP:NS.btnG]}>
    <Text style={kind==='primary'?NS.btnTxP:NS.btnTxG}>{label}</Text>
  </Pressable>;
}

const H = StyleSheet.create({
  bar:{height:56,paddingHorizontal:12,flexDirection:'row',alignItems:'center',
       borderBottomWidth:StyleSheet.hairlineWidth,borderBottomColor:'#eee'},
  back:{padding:6},
  title:{flex:1,textAlign:'center',fontSize:18,fontWeight:'700',color:'#111'},
});

const NS = StyleSheet.create({
  wrap:{flex:1,padding:20,backgroundColor:'#fafafa',gap:16},
  card:{backgroundColor:'#fff',borderRadius:14,padding:16,gap:10,elevation:1,
        shadowColor:'#000',shadowOpacity:0.05,shadowRadius:6},
  row:{flexDirection:'row',justifyContent:'space-between',alignItems:'center',paddingVertical:6},
  label:{color:'#666'}, right:{fontWeight:'600'},
  btn:{paddingVertical:12,borderRadius:12,alignItems:'center'},
  btnP:{backgroundColor:'#191919'}, btnG:{backgroundColor:'#f2f2f2'},
  btnTxP:{color:'#fff',fontWeight:'700'}, btnTxG:{color:'#333',fontWeight:'600'},
  token:{fontSize:12,color:'#666',marginTop:6}, tip:{color:'#777',fontSize:12},
});
