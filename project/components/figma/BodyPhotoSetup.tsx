// project/components/figma/BodyPhotoSetup.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  Pressable,
  ActivityIndicator,
  ScrollView,
  Alert as RNAlert,
  Platform, // Platform import 추가 
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { Camera, Upload, CheckCircle, Smartphone } from 'lucide-react-native';
import AppHeader from '../common/AppHeader';
import BottomNavBar from '../common/BottomNavBar';

export default function BodyPhotoSetup({
  onComplete,
  onBack,
}: {
  onComplete: () => void;
  onBack: () => void;
}) {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  // 웹에서는 이 기능들이 호출되지 않도록 합니다.
  async function pickFromCamera() {
    if (Platform.OS === 'web') return;
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      RNAlert.alert('권한 필요', '카메라 권한을 허용해줘.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      aspect: [3, 4],
      quality: 0.8,
    });
    if (!res.canceled) {
      startAnalysis(res.assets[0].uri);
    }
  }

  async function pickFromLibrary() {
    if (Platform.OS === 'web') return;
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      RNAlert.alert('권한 필요', '사진 라이브러리 권한을 허용해줘.');
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      aspect: [3, 4],
      quality: 0.8,
    });
    if (!res.canceled) {
      startAnalysis(res.assets[0].uri);
    }
  }

  function startAnalysis(uri: string) {
    setImageUri(uri);
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 3000);
  }

  // 웹 전용 안내 UI
  const WebNotSupported = () => (
    <View style={styles.placeholder}>
        <View style={styles.placeholderIcon}>
            <Smartphone size={32} color="#9CA3AF" />
        </View>
        <View style={{ alignItems: 'center', gap: 8 }}>
            <Text style={styles.placeholderMain}>웹에서는 지원하지 않는 기능입니다.</Text>
            <Text style={styles.placeholderSub}>모바일 앱으로 접속하여 이용해주세요.</Text>
        </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.safe}>
      <AppHeader title="체형 분석" subtitle="AI 기반 체형 측정" onBack={onBack} />

      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.card}>
          <View style={styles.alert}>
            <Text style={styles.alertTitle}>최적의 분석을 위한 촬영 가이드</Text>
            <Text style={styles.alertItem}>• 정면을 바라보고 서서 전신이 나오도록 촬영해주세요.</Text>
            <Text style={styles.alertItem}>• 체형이 잘 드러나는 옷을 입고 밝은 곳에서 촬영해주세요.</Text>
          </View>

          {/* Platform.OS를 사용하여 웹과 모바일을 구분 */}
          {Platform.OS === 'web' ? (
            <WebNotSupported />
          ) : (
            <>
              {!imageUri ? (
                <View style={{ gap: 16 }}>
                  <View style={styles.placeholder}>
                    <View style={styles.placeholderIcon}>
                      <Camera size={32} color="#9CA3AF" />
                    </View>
                    <Text style={styles.placeholderMain}>체형 분석을 위한 사진을 업로드해주세요.</Text>
                  </View>
                  <View style={styles.row2}>
                    <Pressable onPress={pickFromCamera} style={styles.btn}>
                      <Camera size={16} color="#111" />
                      <Text style={styles.btnText}>카메라 촬영</Text>
                    </Pressable>
                    <Pressable onPress={pickFromLibrary} style={styles.btn}>
                      <Upload size={16} color="#111" />
                      <Text style={styles.btnText}>사진 선택</Text>
                    </Pressable>
                  </View>
                </View>
              ) : (
                <View style={{ gap: 16 }}>
                  <Image source={{ uri: imageUri }} style={styles.previewBox} />
                  {isAnalyzing && (
                    <View style={{ alignItems: 'center', gap: 12 }}>
                      <ActivityIndicator />
                      <Text>AI가 체형을 분석하고 있습니다...</Text>
                    </View>
                  )}
                  {analysisComplete && (
                    <View style={{ gap: 16 }}>
                        <Text style={{textAlign: 'center'}}>분석 완료! 회원님은 <Text style={{fontWeight: 'bold'}}>애플형</Text> 체형입니다.</Text>
                        <Pressable onPress={onComplete} style={[styles.btn, {backgroundColor: '#111'}]}>
                            <Text style={[styles.btnText, {color: '#FFF'}]}>다음 단계</Text>
                        </Pressable>
                    </View>
                  )}
                </View>
              )}
            </>
          )}
        </View>
      </ScrollView>
      <BottomNavBar disabled />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F9FAFB' },
  container: { padding: 16, flexGrow: 1, justifyContent: 'center' },
  card: { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 16, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 10, elevation: 2 },
  alert: { backgroundColor: '#F3F4F6', padding: 12, borderRadius: 8, marginBottom: 16 },
  alertTitle: { fontSize: 14, fontWeight: '600', color: '#111827' },
  alertItem: { fontSize: 12, color: '#4B5563', marginTop: 4 },
  placeholder: { borderWidth: 1, borderColor: '#E5E7EB', backgroundColor: '#F9FAFB', aspectRatio: 3 / 4, alignItems: 'center', justifyContent: 'center', gap: 12, borderRadius: 8 },
  placeholderIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: '#E5E7EB', alignItems: 'center', justifyContent: 'center' },
  placeholderMain: { fontSize: 14, color: '#4B5563', fontWeight: '600' },
  placeholderSub: { fontSize: 12, color: '#6B7280' },
  row2: { flexDirection: 'row', gap: 12 },
  btn: { flex: 1, paddingVertical: 14, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 8, borderRadius: 8, borderWidth: 1, borderColor: '#E5E7EB' },
  btnText: { color: '#111', fontSize: 14, fontWeight: '600' },
  previewBox: { aspectRatio: 3 / 4, borderRadius: 8, backgroundColor: '#E5E7EB' },
});
