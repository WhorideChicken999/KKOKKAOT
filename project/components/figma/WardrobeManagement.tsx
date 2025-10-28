// components/figma/WardrobeManagement.tsx

import React, { useState, useEffect, useCallback, useMemo } from 'react';  // ✅ useMemo 추가
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TextInput,
  Pressable,
  Alert,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Grid3X3, List, Search, Filter, Heart, MoreVertical } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import AppHeader from '../common/AppHeader';
import BottomNavBar from '../common/BottomNavBar';
import { MainScreen } from '../../App';

type ViewMode = 'grid' | 'list';
type Item = { 
  id: number; 
  name: string; 
  brand: string; 
  image: string; 
  category: string; 
  loved: boolean;
  is_outer?: boolean;  // ✅ 추가
  has_top?: boolean;   // ✅ 추가
  has_bottom?: boolean; // ✅ 추가
  has_outer?: boolean;  // ✅ 추가
  has_dress?: boolean;  // ✅ 추가
  // 카테고리별 이미지 URL
  full_image?: string;
  top_image?: string | null;
  bottom_image?: string | null;
  outer_image?: string | null;
  dress_image?: string | null;
};

// 🌐 API 주소 (ngrok 주소로 변경하세요)
const API_BASE_URL = 'https://loyd-extemporaneous-annalise.ngrok-free.dev';
// const API_BASE_URL = 'http://10.0.2.2:4000';
export default function WardrobeManagement({
  onBack,
  onNavigate,
}: {
  onBack: () => void;
  onNavigate: (step: MainScreen) => void;
}) {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [uploading, setUploading] = useState(false);
  const [items, setItems] = useState<Item[]>([]);
  const [userItems, setUserItems] = useState<Item[]>([]);
  const [defaultItems, setDefaultItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<number | null>(null);
  
  // ✅ 카테고리 필터 추가 (4개 카테고리로 확장)
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'top' | 'bottom' | 'outer' | 'dress'>('all');
  
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [selectedItemDetail, setSelectedItemDetail] = useState<any>(null);

  // const total = items.length;

  // 📥 구분된 옷장 데이터 가져오기 (사용자 아이템 + 기본 아이템)
  const fetchWardrobe = useCallback(async () => {
    if (!userId) {
      console.log('⚠️ userId가 없습니다!');
      return;
    }

    setLoading(true);
    try {
      const url = `${API_BASE_URL}/api/wardrobe/separated/${userId}`;
      console.log('\n========================================');
      console.log('🔍 구분된 옷장 조회 시작');
      console.log('👤 user_id:', userId);
      console.log('🌐 요청 URL:', url);
      console.log('========================================\n');
      
      const response = await fetch(url);
      const text = await response.text();
      console.log('📄 응답 본문 (text):', text.substring(0, 200));

      const data = JSON.parse(text);
      console.log('📦 서버 응답 (JSON):', data);

      if (data.success) {
        console.log('✅ 성공! 사용자 아이템:', data.total_user_items, '기본 아이템:', data.total_default_items);
        
        // 사용자 아이템이 없고 기본 아이템만 있는 경우 알림
        if (data.total_user_items === 0 && data.total_default_items > 0) {
          Alert.alert(
            '기본 아이템 표시 중',
            '아직 등록된 옷이 없어서 추천용 기본 아이템을 보여드립니다.\n\n"추가" 버튼으로 나만의 옷을 등록해보세요!',
            [{ text: '확인' }]
          );
        }
        
        // 사용자 아이템과 기본 아이템을 각각 처리
        const userItemsList: Item[] = data.user_items.map((item: any, index: number) => {
          return createItemFromData(item, userId, true);
        });
        
        const defaultItemsList: Item[] = data.default_items.map((item: any, index: number) => {
          return createItemFromData(item, userId, false);
        });
        
        // 전체 아이템 (필터링용)
        const allItems = [...userItemsList, ...defaultItemsList];
        
        setUserItems(userItemsList);
        setDefaultItems(defaultItemsList);
        setItems(allItems);
        
        console.log('\n✅ 구분된 옷장 로드 완료!');
        console.log('👤 사용자 아이템:', userItemsList.length, '개');
        console.log('📦 기본 아이템:', defaultItemsList.length, '개');
        console.log('========================================\n');
        
        return;
      } else {
        console.error('❌ 서버 응답 실패:', data.message);
      }
    } catch (error) {
      console.error('\n❌❌❌ 옷장 조회 실패 ❌❌❌');
      console.error('에러:', error);
      console.error('========================================\n');
      
      Alert.alert(
        '옷장 조회 실패', 
        `서버: ${API_BASE_URL}\n\n에러: ${error}`
      );
    } finally {
      setLoading(false);
    }
  }, [userId]);

  // 아이템 데이터 변환 헬퍼 함수
  const createItemFromData = (item: any, userId: number, isUserItem: boolean) => {
    let name = '';
    let category = '';
    
    // ✅ 다중 카테고리 지원: 여러 의류가 있는 경우 우선순위 적용
    const categories = [];
    if (item.has_dress) categories.push('dress');
    if (item.has_outer) categories.push('outer');
    if (item.has_top) categories.push('top');
    if (item.has_bottom) categories.push('bottom');
    
    // ✅ 감지된 카테고리들을 조합하여 이름 생성
    if (categories.length === 0) {
      name = `아이템 ${item.item_id}`;
      category = 'all';
    } else if (categories.length === 1) {
      // 단일 카테고리
      const singleCategory = categories[0];
      if (singleCategory === 'dress') {
        name = '드레스';
        category = 'dress';
      } else if (singleCategory === 'outer') {
        name = '아우터';
        category = 'outer';
      } else if (singleCategory === 'top') {
        name = '상의';
        category = 'top';
      } else if (singleCategory === 'bottom') {
        name = '하의';
        category = 'bottom';
      }
    } else {
      // 다중 카테고리 - 감지된 카테고리들을 "/"로 연결
      const categoryNames = categories.map(cat => {
        switch(cat) {
          case 'dress': return '드레스';
          case 'outer': return '아우터';
          case 'top': return '상의';
          case 'bottom': return '하의';
          default: return cat;
        }
      });
      name = categoryNames.join(' / ');
      category = categories[0]; // 첫 번째 카테고리를 메인으로
    }
    
    // ✅ 이미지 URL 생성
    let imageUrl = '';
    
    if (item.image_path) {
      imageUrl = `${API_BASE_URL}${item.image_path}`;
    } else {
      // 폴백 이미지
      imageUrl = `${API_BASE_URL}/api/images/placeholder.jpg`;
    }
    
    return {
      id: item.item_id,
      name: name || '새 아이템',
      brand: item.is_default ? '기본 아이템' : 'My Wardrobe',
      image: imageUrl,
      category: category,
      loved: false,
      has_top: item.has_top,
      has_bottom: item.has_bottom,
      has_outer: item.has_outer,
      has_dress: item.has_dress,
      is_default: item.is_default,
      // 카테고리별 이미지 URL 저장
      full_image: item.image_path ? `${API_BASE_URL}${item.image_path}` : imageUrl,
      top_image: item.top_image ? `${API_BASE_URL}${item.top_image}` : null,
      bottom_image: item.bottom_image ? `${API_BASE_URL}${item.bottom_image}` : null,
      outer_image: item.outer_image ? `${API_BASE_URL}${item.outer_image}` : null,
      dress_image: item.dress_image ? `${API_BASE_URL}${item.dress_image}` : null
    };
  };

  // ✅ 1. 로그인한 사용자 정보 불러오기
  useEffect(() => {
    const loadUser = async () => {
      try {
        const userStr = await AsyncStorage.getItem('@kko/user');
        if (userStr) {
          const user = JSON.parse(userStr);
          setUserId(user.user_id);
        } else {
          Alert.alert('로그인 필요', '로그인이 필요합니다.');
        }
      } catch (error) {
        console.error('❌ 사용자 정보 로드 실패:', error);
      }
    };
    
    loadUser();
  }, []);

  // ✅ 2. userId가 있으면 옷장 데이터 가져오기
  useEffect(() => {
    if (userId) {
      fetchWardrobe();
    }
  }, [userId, fetchWardrobe]);
  
  // 🗑️ 옷장 아이템 삭제 로직 (추가)
  const deleteItem = async (itemId: number, itemName: string) => {
    Alert.alert(
      '아이템 삭제 확인',
      `정말로 "${itemName}"을(를) 옷장에서 삭제하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        { 
          text: '삭제', 
          style: 'destructive',
          onPress: async () => {
            setLoading(true); // 삭제 요청 동안 로딩 표시
            try {
              const url = `${API_BASE_URL}/api/wardrobe/${itemId}`;
              console.log(`\n🗑️ 삭제 요청: ${url}`);
              
              const response = await fetch(url, {
                method: 'DELETE',
              });

              if (response.ok) {
                Alert.alert('삭제 성공', `${itemName}이(가) 옷장에서 삭제되었습니다.`);
                fetchWardrobe(); // 목록 새로고침
              } else {
                const errorData = await response.json();
                Alert.alert('삭제 실패', errorData.detail || '삭제에 실패했습니다. 서버 로그를 확인하세요.');
              }
            } catch (error) {
              console.error('❌ 삭제 네트워크 오류:', error);
              Alert.alert('네트워크 오류', '삭제 중 서버와 연결할 수 없습니다.');
            } finally {
              setLoading(false);
            }
          }
        },
      ]
    );
  };

  // 아이템 상세 정보 표시 함수 (수정)
  const showItemDetail = async (itemId: number) => {
    setLoading(true);
    try {
      const url = `${API_BASE_URL}/api/wardrobe/item/${itemId}`;
      console.log(`\n🔍 상세 정보 요청: ${url}`);
      
      const response = await fetch(url);
      console.log(`📡 상세 정보 응답 상태: ${response.status}`);
      
      const data = await response.json();
      console.log(`📦 상세 정보 응답 데이터:`, data);
      
      if (data.success && data.item) {
        setSelectedItemDetail(data.item);
        setShowDetailModal(true);
      } else {
        console.error('❌ 상세 정보 조회 실패:', data);
        Alert.alert('오류', data.message || '상세 정보를 불러올 수 없습니다.');
      }
    } catch (error) {
      console.error('❌ 상세 정보 조회 실패:', error);
      Alert.alert('네트워크 오류', '서버와 연결할 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 옷장 아이템 옵션 메뉴 표시 (수정)
  const showItemOptions = (item: Item) => {
    Alert.alert(
      item.name,
      '원하는 작업을 선택해주세요.',
      [
        { text: '정보 보기', onPress: () => showItemDetail(item.id) },
        { text: '편집', onPress: () => Alert.alert('알림', '편집 기능은 곧 추가될 예정입니다.') },
        { text: '삭제', style: 'destructive', onPress: () => deleteItem(item.id, item.name) },
        { text: '취소', style: 'cancel' },  // 👈 맨 뒤로 이동 (Android 순서: 정보보기→편집→삭제→취소)
      ],
      { cancelable: true }  // Android에서 바깥 영역 터치로도 닫기 가능
    );
  };
  
  // 📤 이미지 업로드
  const uploadImage = async (imageUri: string) => {
    if (userId === null || userId === undefined) {
      Alert.alert('오류', '사용자 정보를 불러올 수 없습니다. 다시 로그인해주세요.');
      return; // 👈 함수 실행 중단
    }

    setUploading(true);

    try {
      const formData = new FormData();
      
      if (Platform.OS === 'web') {
        const response = await fetch(imageUri);
        const blob = await response.blob();
        const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' });
        formData.append('image', file);
        console.log('📤 웹에서 업로드 시작...');
      } else {
        const filename = imageUri.split(/\\|\//).pop() || 'photo.jpg';
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';

        formData.append('image', {
          uri: imageUri,
          name: filename,
          type: type,
        } as any);
        
        console.log('📤 모바일에서 업로드 시작...');
      }
      
      formData.append('user_id', String(userId)); // 👈 실제 user_id 사용

      console.log(`🌐 업로드 요청 (user_id: ${userId})`);
      
      const uploadResponse = await fetch(`${API_BASE_URL}/api/upload-wardrobe`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
      });

      const data = await uploadResponse.json();
      console.log('📦 업로드 응답:', data);
      console.log('🔍 outer_attributes:', data.outer_attributes);
      console.log('🔍 top_attributes:', data.top_attributes);
      console.log('🔍 bottom_attributes:', data.bottom_attributes);
      console.log('🔍 dress_attributes:', data.dress_attributes);

      if (data.success) {
        // ✅ 상세한 분석 결과 표시 (카테고리별로 구조화)
        let message = `🎉 업로드 성공!\n\n🤖 AI 분석 완료!\n\n`;
        
        // 아우터 분석 결과 (우선순위 1)
        if (data.outer_attributes) {
          message += `🧥 아우터\n`;
          message += `  - 카테고리: ${data.outer_attributes.category?.value || 'N/A'}\n`;
          message += `  - 컬러: ${data.outer_attributes.color?.value || 'N/A'}\n`;
          message += `  - 핏: ${data.outer_attributes.fit?.value || 'N/A'}\n`;
          message += `  - 스타일: ${data.outer_attributes.style?.value || 'N/A'}\n`;
          message += `  - 소재: ${data.outer_attributes.material?.value || 'N/A'}\n`;
          message += `  - 소매: ${data.outer_attributes.sleeve?.value || 'N/A'}\n\n`;
        }
        
        // 상의 분석 결과 (우선순위 2)
        if (data.top_attributes) {
          message += `👕 상의\n`;
          message += `  - 카테고리: ${data.top_attributes.category?.value || 'N/A'}\n`;
          message += `  - 컬러: ${data.top_attributes.color?.value || 'N/A'}\n`;
          message += `  - 핏: ${data.top_attributes.fit?.value || 'N/A'}\n`;
          message += `  - 스타일: ${data.top_attributes.style?.value || 'N/A'}\n`;
          message += `  - 소재: ${data.top_attributes.material?.value || 'N/A'}\n`;
          message += `  - 소매: ${data.top_attributes.sleeve?.value || 'N/A'}\n\n`;
        }
        
        // 하의 분석 결과 (우선순위 3)
        if (data.bottom_attributes) {
          message += `👖 하의\n`;
          message += `  - 카테고리: ${data.bottom_attributes.category?.value || 'N/A'}\n`;
          message += `  - 컬러: ${data.bottom_attributes.color?.value || 'N/A'}\n`;
          message += `  - 핏: ${data.bottom_attributes.fit?.value || 'N/A'}\n`;
          message += `  - 스타일: ${data.bottom_attributes.style?.value || 'N/A'}\n`;
          message += `  - 소재: ${data.bottom_attributes.material?.value || 'N/A'}\n`;
          message += `  - 길이: ${data.bottom_attributes.length?.value || 'N/A'}\n\n`;
        }
        
        // 드레스 분석 결과 (우선순위 4)
        if (data.dress_attributes) {
          message += `👗 드레스\n`;
          message += `  - 카테고리: ${data.dress_attributes.category?.value || 'N/A'}\n`;
          message += `  - 컬러: ${data.dress_attributes.color?.value || 'N/A'}\n`;
          message += `  - 스타일: ${data.dress_attributes.style?.value || 'N/A'}\n`;
          message += `  - 소재: ${data.dress_attributes.material?.value || 'N/A'}\n`;
          message += `  - 프린트: ${data.dress_attributes.print?.value || 'N/A'}\n\n`;
        }
        
        // 아이템 ID 정보
        if (data.item_id) {
          message += `🆔 아이템 ID: ${data.item_id}`;
        }
        
        Alert.alert('업로드 성공!', message, [
          { 
            text: '확인', 
            onPress: () => {
              console.log('🔄 옷장 새로고침...');
              fetchWardrobe();
            }
          }
        ]);
      } else {
        // ✅ 의류 감지 실패 시 특별한 알림
        if (data.error_type === 'detection_failed') {
          Alert.alert(
            '의류가 확인되지 않습니다', 
            `저장되지 않았습니다.\n\n이유) ${data.error_details || '이미지에서 의류를 찾을 수 없습니다.'}\n\n💡 해결 방법:\n• 의류가 명확하게 보이는 사진을 사용해주세요\n• 배경이 복잡하지 않은 사진을 권장합니다\n• 의류가 이미지 중앙에 위치하도록 촬영해주세요`,
            [{ text: '확인' }]
          );
        } else {
          Alert.alert('업로드 실패', data.message || '다시 시도해주세요.');
        }
      }
    } catch (error) {
      console.error('❌ 업로드 에러:', error);
      Alert.alert('에러', `업로드 중 오류가 발생했습니다.\n\n${error}`);
    } finally {
      setUploading(false);
    }
  };

  // 카메라 권한 요청
  const requestPermissions = async () => {
    if (Platform.OS !== 'web') {
      const cameraPermission = await ImagePicker.requestCameraPermissionsAsync();
      const galleryPermission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (cameraPermission.status !== 'granted' || galleryPermission.status !== 'granted') {
        Alert.alert('권한 필요', '카메라와 갤러리 접근 권한이 필요합니다.');
        return false;
      }
    }
    return true;
  };

  // 카메라로 촬영
  const takePhoto = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [3, 4],
      quality: 0.8,
    });

    if (!result.canceled) {
      uploadImage(result.assets[0].uri);
    }
  };

  // 갤러리에서 선택
  const pickImage = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [3, 4],
      quality: 0.8,
    });

    if (!result.canceled) {
      uploadImage(result.assets[0].uri);
    }
  };

  // 옵션 선택
  const showImageOptions = () => {
    if (uploading) return;

    if (Platform.OS === 'web') {
      pickImage();
    } else {
      Alert.alert(
        '사진 선택',
        '어떻게 추가하시겠어요?',
        [
          { text: '📸 카메라로 촬영', onPress: takePhoto },
          { text: '🖼️ 갤러리에서 선택', onPress: pickImage },
          { text: '취소', style: 'cancel' },
        ]
      );
    }
  };
  const filteredItems = useMemo(() => {
    if (selectedCategory === 'all') {
      return items;
    }
    
    return items.filter(item => {
      // 실제 has_* 필드를 기반으로 필터링
      switch (selectedCategory) {
        case 'top':
          return item.has_top === true;
        case 'bottom':
          return item.has_bottom === true;
        case 'outer':
          return item.has_outer === true;
        case 'dress':
          return item.has_dress === true;
        default:
          return false;
      }
    });
  }, [items, selectedCategory]);

  const total = filteredItems.length;

  // ✅ 카테고리별 이미지 선택 함수
  const getImageForCategory = (item: Item, category: string) => {
    switch (category) {
      case 'all':
        return item.full_image || item.image;
      case 'top':
        return item.top_image || item.full_image || item.image;
      case 'bottom':
        return item.bottom_image || item.full_image || item.image;
      case 'outer':
        return item.outer_image || item.full_image || item.image;
      case 'dress':
        return item.dress_image || item.full_image || item.image;
      default:
        return item.image;
    }
  };

  // 헤더 우측 버튼
  const HeaderRightAction = (
    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
      <Pressable onPress={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')} style={styles.iconBtn}>
        {viewMode === 'grid' ? <List size={20} color="#111" /> : <Grid3X3 size={20} color="#111" />}
      </Pressable>
      <Pressable 
        onPress={showImageOptions} 
        style={[styles.addBtn, uploading && styles.addBtnDisabled]}
        disabled={uploading}
      >
        {uploading ? (
          <>
            <ActivityIndicator size="small" color="#FFF" />
            <Text style={styles.addBtnText}>분석중...</Text>
          </>
        ) : (
          <>
            <Camera size={16} color="#FFF" />
            <Text style={styles.addBtnText}>추가</Text>
          </>
        )}
      </Pressable>
    </View>
  );

  // 로딩 화면
  if (loading && items.length === 0) {
    return (
      <SafeAreaView style={styles.safe}>
        <AppHeader title="내 옷장" subtitle="로딩 중..." onBack={onBack} />
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator size="large" color="#111" />
          <Text style={{ marginTop: 16, color: '#666' }}>옷장을 불러오는 중...</Text>
        </View>
        <BottomNavBar activeScreen="wardrobe-management" onNavigate={onNavigate} />
      </SafeAreaView>
    );
  }

  // ✅ 상세 정보 모달 렌더링 함수
  const renderDetailModal = () => {
    if (!showDetailModal || !selectedItemDetail) return null;

    const item = selectedItemDetail;
    
    return (
      <View style={styles.modalOverlay}>
        <Pressable 
          style={styles.modalBackdrop} 
          onPress={() => setShowDetailModal(false)}
        />
        <View style={styles.modalContent}>
          <ScrollView showsVerticalScrollIndicator={false}>
            {/* 모달 헤더 */}
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>AI 분석 결과</Text>
              <Pressable onPress={() => setShowDetailModal(false)}>
                <Text style={styles.modalClose}>✕</Text>
              </Pressable>
            </View>

            {/* 이미지는 표시하지 않음 - AI 예측 결과만 표시 */}




            {/* 상의 정보 */}
            {item.has_top && item.attributes?.top && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>👕 상의</Text>
                
                <View style={styles.modalInfoBox}>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>카테고리</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.category || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>색상</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.color || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>핏</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.fit || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소재</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.material || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>프린트</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.print_pattern || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>스타일</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.style || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소매 길이</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.sleeve_length || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>성별</Text>
                    <Text style={styles.modalValue}>{item.attributes.top.gender || 'N/A'}</Text>
                  </View>
                </View>
              </View>
            )}

            {/* 하의 정보 */}
            {item.has_bottom && item.attributes?.bottom && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>👖 하의</Text>
                
                <View style={styles.modalInfoBox}>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>카테고리</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.category || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>색상</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.color || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>핏</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.fit || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소재</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.material || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>프린트</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.print_pattern || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>스타일</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.style || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>길이</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.length || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>성별</Text>
                    <Text style={styles.modalValue}>{item.attributes.bottom.gender || 'N/A'}</Text>
                  </View>
                </View>
              </View>
            )}

            {/* 아우터 정보 */}
            {item.has_outer && item.attributes?.outer && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>🧥 아우터</Text>
                
                <View style={styles.modalInfoBox}>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>카테고리</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.category || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>색상</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.color || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>핏</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.fit || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소재</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.material || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>프린트</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.print_pattern || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>스타일</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.style || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소매 길이</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.sleeve_length || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>성별</Text>
                    <Text style={styles.modalValue}>{item.attributes.outer.gender || 'N/A'}</Text>
                  </View>
                </View>
              </View>
            )}

            {/* 원피스 정보 */}
            {item.has_dress && item.attributes?.dress && (
              <View style={styles.modalSection}>
                <Text style={styles.modalSectionTitle}>👗 원피스</Text>
                
                <View style={styles.modalInfoBox}>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>카테고리</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.category || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>색상</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.color || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>소재</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.material || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>프린트</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.print_pattern || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>스타일</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.style || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalInfoRow}>
                    <Text style={styles.modalLabel}>성별</Text>
                    <Text style={styles.modalValue}>{item.attributes.dress.gender || 'N/A'}</Text>
                  </View>
                </View>
              </View>
            )}

            {/* 닫기 버튼 */}
            <Pressable 
              style={styles.modalCloseButton} 
              onPress={() => setShowDetailModal(false)}
            >
              <Text style={styles.modalCloseButtonText}>닫기</Text>
            </Pressable>
          </ScrollView>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <AppHeader title="내 옷장" subtitle={`${total}개 아이템`} onBack={onBack} rightAction={HeaderRightAction} />

      <ScrollView contentContainerStyle={styles.screenPad}>
        {/* 검색 바 */}
        <View style={{ position: 'relative', marginBottom: 16 }}>
          <View style={{position: 'absolute', left: 12, top: 14, zIndex: 1}}>
            <Search size={16} color="#9CA3AF" />
          </View>
          <TextInput placeholder="아이템 검색..." style={styles.searchInput} />
          <Pressable style={styles.filterBtn}><Filter size={16} color="#111" /></Pressable>
        </View>

        {/* 카테고리 필터 */}
        <View style={styles.categoryFilter}>
          <Pressable 
            style={[
              styles.categoryBtn, 
              selectedCategory === 'top' && styles.categoryBtnActive
            ]}
            onPress={() => setSelectedCategory('top')}
          >
            <Text style={[
              styles.categoryBtnText,
              selectedCategory === 'top' && styles.categoryBtnTextActive
            ]}>
              👕 상의 ({items.filter(i => i.has_top === true).length})
            </Text>
          </Pressable>

          <Pressable 
            style={[
              styles.categoryBtn, 
              selectedCategory === 'bottom' && styles.categoryBtnActive
            ]}
            onPress={() => setSelectedCategory('bottom')}
          >
            <Text style={[
              styles.categoryBtnText,
              selectedCategory === 'bottom' && styles.categoryBtnTextActive
            ]}>
              👖 하의 ({items.filter(i => i.has_bottom === true).length})
            </Text>
          </Pressable>

          <Pressable 
            style={[
              styles.categoryBtn, 
              selectedCategory === 'outer' && styles.categoryBtnActive
            ]}
            onPress={() => setSelectedCategory('outer')}
          >
            <Text style={[
              styles.categoryBtnText,
              selectedCategory === 'outer' && styles.categoryBtnTextActive
            ]}>
              🧥 아우터 ({items.filter(i => i.has_outer === true).length})
            </Text>
          </Pressable>

          <Pressable 
            style={[
              styles.categoryBtn, 
              selectedCategory === 'dress' && styles.categoryBtnActive
            ]}
            onPress={() => setSelectedCategory('dress')}
          >
            <Text style={[
              styles.categoryBtnText,
              selectedCategory === 'dress' && styles.categoryBtnTextActive
            ]}>
              👗 원피스 ({items.filter(i => i.has_dress === true).length})
            </Text>
          </Pressable>

          <Pressable 
            style={[
              styles.categoryBtn, 
              selectedCategory === 'all' && styles.categoryBtnActive
            ]}
            onPress={() => setSelectedCategory('all')}
          >
            <Text style={[
              styles.categoryBtnText,
              selectedCategory === 'all' && styles.categoryBtnTextActive
            ]}>
              전체 ({items.length})
            </Text>
          </Pressable>
        </View>

        {/* 구분된 옷장 섹션 */}
        {userItems.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionTitle}>👤 내 옷 ({userItems.length}개)</Text>
            <View style={styles.gridWrap}>
              {userItems
                .filter(item => {
                  if (selectedCategory === 'all') return true;
                  switch (selectedCategory) {
                    case 'top': return item.has_top === true;
                    case 'bottom': return item.has_bottom === true;
                    case 'outer': return item.has_outer === true;
                    case 'dress': return item.has_dress === true;
                    default: return false;
                  }
                })
                .map((item, index) => (
                <Pressable key={`user-${item.id}-${index}`} style={styles.card}>
                  <Image 
                    source={{ uri: getImageForCategory(item, selectedCategory) }} 
                    style={styles.cardImg}
                    onError={(e) => console.error('❌ 이미지 로드 실패:', getImageForCategory(item, selectedCategory), e.nativeEvent.error)}
                    onLoad={() => console.log('✅ 이미지 로드 성공:', getImageForCategory(item, selectedCategory))}
                  />
                  <View style={styles.cardTopRight}>
                    {item.loved && <View style={styles.roundBtnWhite}><Heart size={14} color="#EF4444" fill="#EF4444" /></View>}
                    <Pressable style={styles.roundBtnWhite} onPress={() => showItemOptions(item)}>
                      <MoreVertical size={14} color="#111" />
                    </Pressable>
                  </View>
                  <View style={styles.cardBottomOverlay}>
                    <Text style={styles.cardName}>{item.name}</Text>
                    <Text style={styles.cardBrand}>{item.brand}</Text>
                  </View>
                </Pressable>
              ))}
            </View>
          </View>
        )}

        {/* 기본 아이템 섹션 */}
        {defaultItems.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionTitle}>📦 추천 아이템 ({defaultItems.length}개)</Text>
            <Text style={styles.sectionSubtitle}>AI가 추천하는 다양한 스타일의 아이템들</Text>
            <View style={styles.gridWrap}>
              {defaultItems
                .filter(item => {
                  if (selectedCategory === 'all') return true;
                  switch (selectedCategory) {
                    case 'top': return item.has_top === true;
                    case 'bottom': return item.has_bottom === true;
                    case 'outer': return item.has_outer === true;
                    case 'dress': return item.has_dress === true;
                    default: return false;
                  }
                })
                .map((item, index) => (
                <Pressable key={`default-${item.id}-${index}`} style={[styles.card, styles.defaultCard]}>
                  <Image 
                    source={{ uri: getImageForCategory(item, selectedCategory) }} 
                    style={styles.cardImg}
                    onError={(e) => console.error('❌ 이미지 로드 실패:', getImageForCategory(item, selectedCategory), e.nativeEvent.error)}
                    onLoad={() => console.log('✅ 이미지 로드 성공:', getImageForCategory(item, selectedCategory))}
                  />
                  <View style={styles.cardTopRight}>
                    <View style={styles.defaultBadge}>
                      <Text style={styles.defaultBadgeText}>추천</Text>
                    </View>
                  </View>
                  <View style={styles.cardBottomOverlay}>
                    <Text style={styles.cardName}>{item.name}</Text>
                    <Text style={styles.cardBrand}>{item.brand}</Text>
                  </View>
                </Pressable>
              ))}
            </View>
          </View>
        )}

        {/* 빈 상태 */}
        {userItems.length === 0 && defaultItems.length === 0 && (
          <View style={{ paddingVertical: 60, alignItems: 'center' }}>
            <Text style={{ fontSize: 16, color: '#9CA3AF', marginBottom: 8 }}>
              {selectedCategory === 'all' 
                ? '아직 옷이 없어요' 
                : selectedCategory === 'top' ? '상의가 없어요'
                : selectedCategory === 'bottom' ? '하의가 없어요'
                : selectedCategory === 'outer' ? '아우터가 없어요'
                : selectedCategory === 'dress' ? '드레스가 없어요'
                : '해당 카테고리 옷이 없어요'}
            </Text>
            <Text style={{ fontSize: 14, color: '#D1D5DB' }}>우측 상단 "추가" 버튼을 눌러 옷을 등록해보세요!</Text>
          </View>
        )}
      </ScrollView>
      
      <BottomNavBar activeScreen="wardrobe-management" onNavigate={onNavigate} />
      {renderDetailModal()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#FFFFFF' },
  iconBtn: { padding: 8 },
  addBtn: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    gap: 6, 
    backgroundColor: '#111', 
    paddingHorizontal: 12, 
    paddingVertical: 8, 
    borderRadius: 8 
  },
  addBtnDisabled: {
    opacity: 0.6,
  },

  // ✅ 카테고리 필터 스타일 추가
  categoryFilter: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
    flexWrap: 'wrap',
  },
  categoryBtn: {
    flex: 1,
    minWidth: '22%',
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: '#F9FAFB',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    alignItems: 'center',
  },
  categoryBtnActive: {
    backgroundColor: '#111',
    borderColor: '#111',
  },
  categoryBtnText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6B7280',
  },
  categoryBtnTextActive: {
    color: '#FFF',
  },

  addBtnText: { color: '#FFF', fontSize: 13, fontWeight: '600' },
  screenPad: { padding: 16, paddingBottom: 24 },
  searchInput: { 
    backgroundColor: '#F9FAFB', 
    borderWidth: 1, 
    borderColor: '#E5E7EB', 
    borderRadius: 8, 
    paddingLeft: 40, 
    paddingRight: 50, 
    paddingVertical: 10, 
    fontSize: 14 
  },
  filterBtn: { 
    position: 'absolute', 
    right: 6, 
    top: 6, 
    padding: 8, 
    backgroundColor: '#FFF', 
    borderWidth: 1, 
    borderColor: '#E5E7EB', 
    borderRadius: 6 
  },
  gridWrap: { 
    flexDirection: 'row', 
    flexWrap: 'wrap', 
    justifyContent: 'space-between', 
    rowGap: 12 
  },
  card: { 
    width: '48.5%', 
    aspectRatio: 3 / 4, 
    borderRadius: 8, 
    overflow: 'hidden', 
    backgroundColor: '#EEE', 
    elevation: 2, 
    shadowColor: '#000', 
    shadowOpacity: 0.1, 
    shadowRadius: 5
  },
  cardImg: { width: '100%', height: '100%' },
  cardTopRight: { 
    position: 'absolute', 
    top: 8, 
    right: 8, 
    flexDirection: 'row', 
    gap: 6 
  },
  roundBtnWhite: { 
    width: 28, 
    height: 28, 
    borderRadius: 14, 
    backgroundColor: 'rgba(255,255,255,0.9)', 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  cardBottomOverlay: { 
    position: 'absolute', 
    bottom: 0, 
    left: 0, 
    right: 0, 
    padding: 8, 
    backgroundColor: 'rgba(0,0,0,0.45)' 
  },
  cardName: { color: '#FFF', fontWeight: '600' },
  cardBrand: { color: '#FFF', fontSize: 11, opacity: 0.9 },
  modalOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1000,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    width: '90%',
    maxHeight: '80%',
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111',
  },
  modalClose: {
    fontSize: 24,
    color: '#6B7280',
    fontWeight: '300',
  },
  modalSection: {
    marginBottom: 20,
  },
  modalSectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111',
    marginBottom: 12,
  },
  modalOriginalImage: {
    width: '100%',
    height: 200,
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
  },
  modalSeparatedImage: {
    width: '100%',
    height: 150,
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
    marginBottom: 12,
  },
  modalInfoBox: {
    backgroundColor: '#F9FAFB',
    borderRadius: 8,
    padding: 12,
    gap: 8,
  },
  modalInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modalLabel: {
    fontSize: 13,
    color: '#6B7280',
    fontWeight: '500',
  },
  modalValue: {
    fontSize: 13,
    color: '#111',
    fontWeight: '600',
  },
  modalCloseButton: {
    backgroundColor: '#111',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  modalCloseButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  
  // ✅ 구분된 옷장 섹션 스타일
  sectionContainer: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111',
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 16,
  },
  defaultCard: {
    borderWidth: 2,
    borderColor: '#E5E7EB',
  },
  defaultBadge: {
    backgroundColor: '#3B82F6',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  defaultBadgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '600',
  },
});
