import React from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Linking } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronLeft } from 'lucide-react-native';

const FAQ_URL = 'https://your-domain.com/faq'; // ← 실제 주소
const SUPPORT_MAIL = 'mailto:support@your-domain.com?subject=%5B꼬까옷%5D%20앱%20문의';

type Props = { onBack?: () => void };

export default function SupportScreen({ onBack }: Props) {
  return (
    <View style={CS.wrap}>
      {onBack && (
        <SafeAreaView edges={['top']} style={{ backgroundColor:'#fff' }}>
          <View style={H.bar}>
            <Pressable onPress={onBack} hitSlop={12} style={H.back}>
              <ChevronLeft size={24} color="#111" />
            </Pressable>
            <Text style={H.title}>고객센터</Text>
            <View style={{ width: 36 }} />
          </View>
        </SafeAreaView>
      )}

      <View style={CS.card}>
        <Item label="자주 묻는 질문(FAQ)" onPress={()=>Linking.openURL(FAQ_URL)} />
        <Item label="이메일 문의하기" onPress={()=>Linking.openURL(SUPPORT_MAIL)} />
        <Item label="채팅 상담 (준비중)" onPress={()=>Alert.alert('곧 제공 예정입니다.')} />
      </View>

      <View style={CS.meta}>
        <Text style={CS.metaTx}>앱 버전 1.0.0</Text>
        <Text style={CS.metaTx}>© 2025 KKO</Text>
      </View>
    </View>
  );
}

function Item({ label, onPress }: any){
  return <Pressable onPress={onPress} style={CS.item}><Text style={CS.itemTx}>{label}</Text></Pressable>;
}

const H = StyleSheet.create({
  bar:{height:56,paddingHorizontal:12,flexDirection:'row',alignItems:'center',
       borderBottomWidth:StyleSheet.hairlineWidth,borderBottomColor:'#eee'},
  back:{padding:6},
  title:{flex:1,textAlign:'center',fontSize:18,fontWeight:'700',color:'#111'},
});

const CS = StyleSheet.create({
  wrap:{flex:1,padding:20,backgroundColor:'#fafafa',gap:16},
  card:{backgroundColor:'#fff',borderRadius:14,paddingVertical:4,elevation:1,
        shadowColor:'#000',shadowOpacity:0.05,shadowRadius:6},
  item:{paddingVertical:16,paddingHorizontal:16,borderBottomColor:'#eee',borderBottomWidth:1},
  itemTx:{fontSize:16},
  meta:{alignItems:'center',gap:4,marginTop:8},
  metaTx:{color:'#777',fontSize:12},
});
