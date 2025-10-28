// components/figma/LLMChatScreen.tsx

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
  ActivityIndicator,
  Alert,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  MessageCircle,
  Send,
  X,
  ArrowLeft,
  Camera,
  Check,
} from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import AppHeader from '../common/AppHeader';
import BottomNavBar from '../common/BottomNavBar';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { MainScreen } from '../../App';

const API_BASE_URL = 'https://loyd-extemporaneous-annalise.ngrok-free.dev';
const BOTTOM_NAV_HEIGHT = 80;

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

type WardrobeItem = {
  id: number;
  name?: string;
  brand?: string;
  category?: string;
  color?: string;
  fit?: string;
  materials?: string[];
  image?: string;
  top_category?: string;
  bottom_category?: string;
  outer_category?: string;
  dress_category?: string;
  top_image?: string;
  bottom_image?: string;
  outer_image?: string;
  dress_image?: string;
  has_top?: boolean;
  has_bottom?: boolean;
  has_outer?: boolean;
  has_dress?: boolean;
  image_path?: string;
  is_default?: boolean;

  // UI 전용 플래그
  is_recommended?: boolean;
  is_selected?: boolean;
};

export default function LLMChatScreen({
  onBack,
  onNavigate,
}: {
  onBack: () => void;
  onNavigate: (step: MainScreen) => void;
}) {
  const [userId, setUserId] = useState<number | null>(null);

  // 내 전체 옷장 (기본템 포함)
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);

  // 채팅 메세지들
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  // 입력창
  const [chatInput, setChatInput] = useState('');

  // 로딩 상태
  const [chatLoading, setChatLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  // 추천 / 선택 아이템 카드들 (UI에 가로 스크롤로 뿌리는 영역)
  const [chatRecommendations, setChatRecommendations] = useState<WardrobeItem[]>([]);

  // 지금 유저가 카드 눌러서 선택해둔 아이템들의 id (한 번에 1개만 유지하도록 설계했었음)
  const [selectedItemIds, setSelectedItemIds] = useState<number[]>([]);

  // --------------------------------------------------
  // 1) 사용자 ID 불러오기 (앱 로드 시 1번만)
  // --------------------------------------------------
  useEffect(() => {
    const loadUserId = async () => {
      console.log('💾 사용자 정보 로딩 시작...');
      try {
        const userData = await AsyncStorage.getItem('@kko/user');
        console.log('📦 AsyncStorage 데이터:', userData);
        if (userData) {
          const user = JSON.parse(userData);
          const resolvedId = user.id || user.user_id;
          console.log('👤 파싱된 사용자 ID:', resolvedId);
          setUserId(resolvedId);
        } else {
          console.log('⚠️ AsyncStorage에 사용자 정보 없음');
        }
      } catch (error) {
        console.error('❌ 사용자 ID 로드 실패:', error);
      }
    };
    loadUserId();
  }, []);

  // --------------------------------------------------
  // 2) 옷장 불러오기 (userId 생기면)
  //    👉 include_defaults=true 로 바꿈 (기본템도 카드로 보여줄 수 있게)
  // --------------------------------------------------
  const fetchWardrobe = useCallback(async () => {
    console.log('👕 옷장 데이터 로딩 시작... userId:', userId);
    if (!userId) {
      console.log('⚠️ userId 없음 - 옷장 데이터 로드 취소');
      return;
    }

    try {
      const url = `${API_BASE_URL}/api/wardrobe/${userId}?include_defaults=true`;
      console.log('📡 API 호출:', url);
      const response = await fetch(url);
      console.log('📥 응답 상태:', response.status);

      if (!response.ok) {
        console.error('❌ 옷장 데이터 로드 실패:', response.status);
        return;
      }

      const data = await response.json();
      console.log('✅ 옷장 데이터 로드 성공:', data.items?.length, '개');

      // 혹시 중복 item.id 있으면 uniq 처리
      const uniqueItems = data.items.filter(
        (item: WardrobeItem, index: number, self: WardrobeItem[]) =>
          index === self.findIndex((t: WardrobeItem) => t.id === item.id),
      );

      console.log('🔄 중복 제거 후:', uniqueItems.length, '개');
      setWardrobeItems(uniqueItems);
    } catch (error) {
      console.error('❌ 옷장 데이터 로드 실패:', error);
    }
  }, [userId]);

  useEffect(() => {
    if (userId) {
      fetchWardrobe();
    }
  }, [userId, fetchWardrobe]);

  // --------------------------------------------------
  // 3) 아이템 카드 선택 토글 (추천 카드 탭하면 선택 / 해제)
  //    - 지금은 1개만 선택 유지
  // --------------------------------------------------
  const toggleItemSelection = (itemId: number) => {
    setSelectedItemIds(prev => {
      if (prev.includes(itemId)) {
        return []; // 이미 선택된 거 또 누르면 비우기
      } else {
        return [itemId]; // 새 선택은 덮어쓰기
      }
    });
  };

  // --------------------------------------------------
  // 4) 채팅 보내기: LLM 호출
  //    백엔드에서 recommendations = [item_id, ...] 형식으로 온다고 가정하고
  //    그걸 wardrobeItems에서 찾아서 카드로 만들어 붙임
  // --------------------------------------------------
  const sendChatMessage = async () => {
    console.log('\n🚀 sendChatMessage 호출됨!');
    console.log('📝 입력값:', chatInput);
    console.log('👤 userId:', userId);
    console.log('👕 선택된 아이템 IDs:', selectedItemIds);
    console.log('⏳ chatLoading:', chatLoading);

    if (!chatInput.trim() || !userId || chatLoading) {
      console.log('⚠️ 조건 실패 - 메시지 전송 취소');
      return;
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date(),
    };

    // 채팅창에 내가 쓴 말 먼저 추가
    console.log('✅ 사용자 메시지 생성:', userMessage.content);
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      // 👉 백엔드 명세: multipart/form-data 로 보내는 중
      const formData = new FormData();
      formData.append('user_id', userId.toString());
      formData.append('message', userMessage.content);

      if (selectedItemIds.length > 0) {
        formData.append('selected_items', JSON.stringify(selectedItemIds));
        console.log('✅ 선택된 아이템 포함:', selectedItemIds);
      }

      console.log('📡 API 요청 시작:', `${API_BASE_URL}/api/chat/recommend`);
      const response = await fetch(`${API_BASE_URL}/api/chat/recommend`, {
        method: 'POST',
        body: formData,
      });
      console.log('📥 API 응답 상태:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      console.log('📦 전체 응답 데이터:', data);
      console.log('🎯 추천 아이템 수(raw):', data.recommendations?.length || 0);
      console.log('🎯 추천 아이템 샘플(raw):', data.recommendations?.[0]);

      // 일단 AI 답변(말풍선) push
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, assistantMessage]);

      // ---------------------------
      // 추천 아이템 카드 만들기
      // ---------------------------

      // 백엔드가 보내는 건 숫자 배열(아이템 id들)일 거라고 가정
      // e.g. [10, 7, 22]
      const recIds: number[] = Array.isArray(data.recommendations)
        ? data.recommendations
        : [];

      // 추천된 id -> 실제 wardrobeItems에서 해당 아이템 정보 찾아오기
      const recItemsDetailed: WardrobeItem[] = recIds
        .map(id => wardrobeItems.find(w => w.id === id))
        .filter((itm): itm is WardrobeItem => !!itm)
        .map(itm => {
          // 이름 만들기 (카테고리 기반으로 사람이 읽을만하게)
          let label = itm.name;
          if (!label) {
            const cats: string[] = [];
            if (itm.has_dress) cats.push('원피스');
            if (itm.has_outer) cats.push('아우터');
            if (itm.has_top) cats.push('상의');
            if (itm.has_bottom) cats.push('하의');
            label = cats.length > 0 ? cats.join(' / ') : `아이템 ${itm.id}`;
          }

          return {
            ...itm,
            name: label,
            brand: itm.is_default ? '기본템' : '내 옷',
            is_recommended: true,
            is_selected: false,
          };
        });

      // 선택된 아이템들(내가 고른 것들)도 같이 카드 상단에 보여줄 건데
      // selectedItemIds 기준으로 wardrobeItems에서 찾아서 붙여줌
      const selectedDetailed: WardrobeItem[] = selectedItemIds
        .map(id => wardrobeItems.find(w => w.id === id))
        .filter((itm): itm is WardrobeItem => !!itm)
        .map(itm => {
          let label = itm.name;
          if (!label) {
            const cats: string[] = [];
            if (itm.has_dress) cats.push('원피스');
            if (itm.has_outer) cats.push('아우터');
            if (itm.has_top) cats.push('상의');
            if (itm.has_bottom) cats.push('하의');
            label = cats.length > 0 ? cats.join(' / ') : `아이템 ${itm.id}`;
          }

          return {
            ...itm,
            name: label,
            brand: itm.is_default ? '기본템' : '내 옷',
            is_recommended: false,
            is_selected: true,
          };
        });

      console.log('🎨 변환된 추천 아이템:', recItemsDetailed);
      console.log('📌 현재 선택 아이템 카드:', selectedDetailed);

      // 다음 턴을 위해 선택은 비워 줌 (UX: 추천 받고 나면 초기화)
      setSelectedItemIds([]);

      // 화면에 뿌릴 카드 리스트 만들기
      // - "선택한 옷" 섹션 (is_selected=true)
      // - "추천 코디" 섹션 (is_recommended=true)
      if (selectedDetailed.length === 0 && recItemsDetailed.length === 0) {
        console.log('⚠️ 추천 아이템이 없음');
        setChatRecommendations([]);
      } else {
        setChatRecommendations([...selectedDetailed, ...recItemsDetailed]);
      }
    } catch (error) {
      console.error('❌ LLM 채팅 실패:', error);
      Alert.alert('오류', 'AI와의 대화 중 오류가 발생했습니다.');

      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.',
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  // --------------------------------------------------
  // 5) 초기 인사 메시지 (맨 처음 화면 들어왔을 때 1번만)
  // --------------------------------------------------
  useEffect(() => {
    console.log('💬 초기 메시지 체크...');
    console.log('  - 옷장 아이템 수:', wardrobeItems.length);
    console.log('  - 채팅 메시지 수:', chatMessages.length);

    if (chatMessages.length === 0 && userId) {
      console.log('✅ 초기 인사 메시지 생성');
      const welcomeMessage: ChatMessage = {
        role: 'assistant',
        content:
          wardrobeItems.length > 0
            ? `안녕하세요! 저는 당신의 패션 스타일리스트 AI입니다. 옷장에 ${wardrobeItems.length}개의 아이템이 있네요. 어떤 스타일링을 도와드릴까요?`
            : `안녕하세요! 저는 당신의 패션 스타일리스트 AI입니다. 어떤 스타일링을 도와드릴까요?`,
        timestamp: new Date(),
      };
      setChatMessages([welcomeMessage]);
    }
  }, [wardrobeItems, chatMessages.length, userId]);

  // --------------------------------------------------
  // 6) 카메라/갤러리 권한
  // --------------------------------------------------
  const requestPermissions = async () => {
    if (Platform.OS === 'web') return true;

    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: libraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (cameraStatus !== 'granted' || libraryStatus !== 'granted') {
      Alert.alert('권한 필요', '카메라 및 갤러리 접근 권한이 필요합니다.');
      return false;
    }
    return true;
  };

  // 촬영
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

  // 갤러리
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

  // --------------------------------------------------
  // 7) 이미지 업로드 -> 백엔드가 새 아이템 분석/등록
  // --------------------------------------------------
  const uploadImage = async (imageUri: string) => {
    if (!userId) {
      Alert.alert('오류', '사용자 정보를 불러올 수 없습니다.');
      return;
    }

    setUploading(true);
    setChatLoading(true);

    // 업로드중이라고 채팅에 띄워놓기
    const uploadingMessage: ChatMessage = {
      role: 'assistant',
      content: '📸 사진 분석 중입니다...',
      timestamp: new Date(),
    };
    setChatMessages(prev => [...prev, uploadingMessage]);

    try {
      const formData = new FormData();

      if (Platform.OS === 'web') {
        const res = await fetch(imageUri);
        const blob = await res.blob();
        const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' });
        formData.append('image', file);
      } else {
        const filename = imageUri.split(/\\|\//).pop() || 'photo.jpg';
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';

        formData.append('image', {
          uri: imageUri,
          name: filename,
          type: type,
        } as any);
      }

      formData.append('user_id', String(userId));

      console.log('📤 이미지 업로드 시작:', `${API_BASE_URL}/api/chat/upload`);

      const uploadResponse = await fetch(`${API_BASE_URL}/api/chat/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          Accept: 'application/json',
        },
      });

      const data = await uploadResponse.json();
      console.log('📦 업로드 응답:', data);

      // "📸 사진 분석 중입니다..." 메시지 지우기
      setChatMessages(prev =>
        prev.filter(msg => msg.content !== '📸 사진 분석 중입니다...'),
      );

      if (data.success) {
        // AI 메시지
        const aiMessage: ChatMessage = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
        };
        setChatMessages(prev => [...prev, aiMessage]);

        // 업로드된 아이템이 있으면 카드에 띄워주기
        if (data.uploaded_item) {
          // uploaded_item은 서버에서 이미 예쁘게 만들어 줄 수도 있고
          // 아니라도 최소한 image_path 같은 건 있을 거라 가정
          const justUploadedCard: WardrobeItem = {
            ...data.uploaded_item,
            is_recommended: false,
            is_selected: true,
          };
          setChatRecommendations([justUploadedCard]);
        }

        // 옷장 다시 불러와서 상태 sync
        fetchWardrobe();
      } else {
        const errorMessage: ChatMessage = {
          role: 'assistant',
          content: data.message || '업로드에 실패했습니다.',
          timestamp: new Date(),
        };
        setChatMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('❌ 업로드 실패:', error);

      // 업로드중 메시지 제거
      setChatMessages(prev =>
        prev.filter(msg => msg.content !== '📸 사진 분석 중입니다...'),
      );

      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: '업로드 중 오류가 발생했습니다. 다시 시도해주세요.',
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setUploading(false);
      setChatLoading(false);
    }
  };

  // --------------------------------------------------
  // 8) 사진 아이콘 눌렀을 때
  // --------------------------------------------------
  const showImageOptions = () => {
    if (uploading) return;

    if (Platform.OS === 'web') {
      pickImage();
    } else {
      Alert.alert('사진 추가', '어떻게 추가하시겠어요?', [
        { text: '📸 카메라로 촬영', onPress: takePhoto },
        { text: '🖼️ 갤러리에서 선택', onPress: pickImage },
        { text: '취소', style: 'cancel' },
      ]);
    }
  };

  // --------------------------------------------------
  // 9) 렌더
  // --------------------------------------------------
  // 추천/선택 카드들 중에서 어떤 섹션을 보여줄지 결정
  const hasSelectedCards = chatRecommendations.some(item => item.is_selected);
  const hasRecommendedCards = chatRecommendations.some(item => item.is_recommended);
  const showWardrobePlain =
    chatRecommendations.length > 0 &&
    !hasSelectedCards &&
    !hasRecommendedCards;

  return (
    <SafeAreaView style={styles.safe}>
      <AppHeader
        title="AI 스타일리스트"
        onBack={onBack}
        rightAction={
          <Pressable style={styles.headerBtn}>
            <MessageCircle size={20} color="#111" />
          </Pressable>
        }
      />

      <KeyboardAvoidingView
        style={styles.container}
        behavior="padding"
        keyboardVerticalOffset={-60}
      >
        {/* 채팅 메시지 영역 */}
        <ScrollView
          style={styles.chatArea}
          contentContainerStyle={styles.chatContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {chatMessages.map((message, index) => (
            <View
              key={index}
              style={[
                styles.messageContainer,
                message.role === 'user' ? styles.userMessage : styles.assistantMessage,
              ]}
            >
              <Text
                style={[
                  styles.messageText,
                  message.role === 'user'
                    ? styles.userMessageText
                    : styles.assistantMessageText,
                ]}
              >
                {message.content}
              </Text>
              <Text style={styles.messageTime}>
                {message.timestamp.toLocaleTimeString('ko-KR', {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </Text>
            </View>
          ))}

          {chatLoading && (
            <View style={[styles.messageContainer, styles.assistantMessage]}>
              <ActivityIndicator size="small" color="#6B7280" />
              <Text
                style={[
                  styles.messageText,
                  styles.assistantMessageText,
                  { marginLeft: 8 },
                ]}
              >
                AI가 답변을 준비 중입니다...
              </Text>
            </View>
          )}
        </ScrollView>

        {/* 📌 선택한 옷 섹션 */}
        {hasSelectedCards && (
          <View style={styles.recommendationsContainer}>
            <View style={styles.recommendationsHeader}>
              <Text style={styles.recommendationsTitle}>📌 선택한 옷</Text>
              <Pressable
                style={styles.closeButton}
                onPress={() => setChatRecommendations([])}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <X size={18} color="#6B7280" />
              </Pressable>
            </View>

            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.recommendationsList}>
                {chatRecommendations
                  .filter(item => item.is_selected)
                  .map((item, index) => (
                    <Pressable
                      key={`selected-${item.id}-${index}`}
                      style={[styles.recommendationCard, styles.selectedItemCard]}
                      onPress={() => {}}
                    >
                      <Image
                        source={{
                          uri: `${API_BASE_URL}${item.image_path || item.image}`,
                        }}
                        style={styles.recommendationImage}
                        onError={e =>
                          console.error(
                            '❌ 이미지 로드 실패:',
                            `${API_BASE_URL}${item.image_path || item.image}`,
                            e.nativeEvent.error,
                          )
                        }
                        onLoad={() =>
                          console.log(
                            '✅ 이미지 로드 성공:',
                            `${API_BASE_URL}${item.image_path || item.image}`,
                          )
                        }
                      />
                      <View style={styles.selectedItemBadge}>
                        <Text style={styles.selectedItemBadgeText}>선택함</Text>
                      </View>
                      <Text style={styles.recommendationName} numberOfLines={2}>
                        {item.name || item.category || '의류'}
                      </Text>
                    </Pressable>
                  ))}
              </View>
            </ScrollView>
          </View>
        )}

        {/* ✨ 추천 코디 섹션 */}
        {hasRecommendedCards && (
          <View style={styles.recommendationsContainer}>
            <View style={styles.recommendationsHeader}>
              <Text style={styles.recommendationsTitle}>✨ 추천 코디</Text>

              {!hasSelectedCards && (
                <Pressable
                  style={styles.closeButton}
                  onPress={() => setChatRecommendations([])}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                  <X size={18} color="#6B7280" />
                </Pressable>
              )}
            </View>

            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.recommendationsList}>
                {chatRecommendations
                  .filter(item => item.is_recommended)
                  .map((item, index) => {
                    // 지금 선택 중인지 (체크마크 표시용)
                    const isCurrentlySelected = selectedItemIds.includes(item.id);

                    return (
                      <Pressable
                        key={`recommended-${item.id}-${index}`}
                        style={[
                          styles.recommendationCard,
                          isCurrentlySelected && styles.recommendationCardSelected,
                        ]}
                        onPress={() => toggleItemSelection(item.id)}
                      >
                        <Image
                          source={{
                            uri: `${API_BASE_URL}${item.image_path || item.image}`,
                          }}
                          style={styles.recommendationImage}
                          onError={e =>
                            console.error(
                              '❌ 이미지 로드 실패:',
                              `${API_BASE_URL}${item.image_path || item.image}`,
                              e.nativeEvent.error,
                            )
                          }
                          onLoad={() =>
                            console.log(
                              '✅ 이미지 로드 성공:',
                              `${API_BASE_URL}${item.image_path || item.image}`,
                            )
                          }
                        />
                        <View style={styles.recommendedBadge}>
                          <Text style={styles.recommendedBadgeText}>추천</Text>
                        </View>

                        {isCurrentlySelected && (
                          <View style={styles.selectedBadge}>
                            <Check size={16} color="#FFF" />
                          </View>
                        )}

                        <Text style={styles.recommendationName} numberOfLines={2}>
                          {item.name || item.category || '의류'}
                        </Text>
                      </Pressable>
                    );
                  })}
              </View>
            </ScrollView>
          </View>
        )}

        {/* 👗 그냥 카드 리스트만 있을 때 (ex. 옷장 전체 보여줘 등) */}
        {showWardrobePlain && (
          <View style={styles.recommendationsContainer}>
            <View style={styles.recommendationsHeader}>
              <Text style={styles.recommendationsTitle}>👗 내 옷장</Text>
              <Pressable
                style={styles.closeButton}
                onPress={() => setChatRecommendations([])}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <X size={18} color="#6B7280" />
              </Pressable>
            </View>

            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.recommendationsList}>
                {chatRecommendations.map((item, index) => {
                  const isCurrentlySelected = selectedItemIds.includes(item.id);

                  return (
                    <Pressable
                      key={`wardrobe-${item.id}-${index}`}
                      style={[
                        styles.recommendationCard,
                        isCurrentlySelected && styles.recommendationCardSelected,
                      ]}
                      onPress={() => toggleItemSelection(item.id)}
                    >
                      <Image
                        source={{
                          uri: `${API_BASE_URL}${item.image_path || item.image}`,
                        }}
                        style={styles.recommendationImage}
                        onError={e =>
                          console.error(
                            '❌ 이미지 로드 실패:',
                            `${API_BASE_URL}${item.image_path || item.image}`,
                            e.nativeEvent.error,
                          )
                        }
                        onLoad={() =>
                          console.log(
                            '✅ 이미지 로드 성공:',
                            `${API_BASE_URL}${item.image_path || item.image}`,
                          )
                        }
                      />

                      {isCurrentlySelected && (
                        <View style={styles.selectedBadge}>
                          <Check size={16} color="#FFF" />
                        </View>
                      )}

                      <Text style={styles.recommendationName} numberOfLines={2}>
                        {item.name || item.category || '의류'}
                      </Text>
                    </Pressable>
                  );
                })}
              </View>
            </ScrollView>
          </View>
        )}

        {/* 입력 영역 */}
        <View style={styles.inputContainer}>
          <Pressable
            style={[styles.imageButton, uploading && styles.imageButtonDisabled]}
            onPress={showImageOptions}
            disabled={uploading}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            {uploading ? (
              <ActivityIndicator size="small" color="#6B7280" />
            ) : (
              <Camera size={20} color="#6B7280" />
            )}
          </Pressable>

          <TextInput
            style={styles.textInput}
            placeholder="AI에게 패션 조언을 요청해보세요..."
            value={chatInput}
            onChangeText={setChatInput}
            multiline
            maxLength={500}
            placeholderTextColor="#9CA3AF"
            editable={!uploading}
          />

          <Pressable
            style={[
              styles.sendButton,
              (!chatInput.trim() || chatLoading || uploading) &&
                styles.sendButtonDisabled,
            ]}
            onPress={() => {
              console.log('🔘 보내기 버튼 클릭됨!');
              sendChatMessage();
            }}
            disabled={!chatInput.trim() || chatLoading || uploading}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            {chatLoading ? (
              <ActivityIndicator size="small" color="#FFF" />
            ) : (
              <Send size={20} color="#FFF" />
            )}
          </Pressable>
        </View>
      </KeyboardAvoidingView>

      <BottomNavBar activeScreen="llm-chat" onNavigate={onNavigate} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  container: {
    flex: 1,
    paddingHorizontal: 16,
  },
  headerBtn: {
    padding: 8,
  },
  chatArea: {
    flex: 1,
    marginTop: 16,
  },
  chatContent: {
    paddingBottom: 24,
    paddingTop: 8,
  },
  messageContainer: {
    marginVertical: 4,
    maxWidth: '80%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#111',
    borderRadius: 18,
    borderBottomRightRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#FFF',
    borderRadius: 18,
    borderBottomLeftRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  messageText: {
    fontSize: 14,
    lineHeight: 20,
  },
  userMessageText: {
    color: '#FFF',
  },
  assistantMessageText: {
    color: '#111',
  },
  messageTime: {
    fontSize: 11,
    color: '#9CA3AF',
    marginTop: 4,
    textAlign: 'right',
  },

  recommendationsContainer: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16, // 섹션 간 간격
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  recommendationsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  recommendationsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111',
  },
  closeButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recommendationsList: {
    flexDirection: 'row',
    gap: 12,
  },
  recommendationCard: {
    width: 80,
    alignItems: 'center',
    position: 'relative',
  },
  recommendationCardSelected: {
    transform: [{ scale: 0.95 }],
  },
  selectedItemCard: {
    borderWidth: 2,
    borderColor: '#3B82F6',
    borderRadius: 8,
    padding: 2,
  },
  recommendationImage: {
    width: 80,
    height: 100,
    borderRadius: 8,
    backgroundColor: '#F3F4F6',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedBadge: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#FFF',
  },
  selectedItemBadge: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: '#3B82F6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
  },
  selectedItemBadgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '600',
  },
  recommendedBadge: {
    position: 'absolute',
    top: 4,
    left: 4,
    backgroundColor: '#10B981',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
  },
  recommendedBadgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '600',
  },
  recommendationName: {
    fontSize: 12,
    color: '#6B7280',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 16,
  },

  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginBottom: BOTTOM_NAV_HEIGHT + 8, // 하단 네비 위로 띄우기
    borderWidth: 1,
    borderColor: '#E5E7EB',
    gap: 12,
  },
  imageButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F3F4F6',
    flexShrink: 0,
  },
  imageButtonDisabled: {
    opacity: 0.5,
  },
  textInput: {
    flex: 1,
    fontSize: 14,
    color: '#111',
    maxHeight: 100,
    minHeight: 20,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    elevation: 2, // Android shadow
    shadowColor: '#000', // iOS shadow
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  sendButtonDisabled: {
    backgroundColor: '#D1D5DB',
  },
});
