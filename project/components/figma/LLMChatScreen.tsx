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
  ImageIcon,
  Check,
} from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import AppHeader from '../common/AppHeader';
import BottomNavBar from '../common/BottomNavBar';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { MainScreen } from '../../App';

const API_BASE_URL = 'https://loyd-extemporaneous-annalise.ngrok-free.dev';
const APP_HEADER_HEIGHT = 56;
const BOTTOM_NAV_HEIGHT = 80;

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

type WardrobeItem = {
  id: number;
  name: string;
  brand: string;
  category: string;
  color: string;
  fit: string;
  materials: string[];
  image: string;
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
  is_recommended?: boolean;  // 추천된 아이템
  is_selected?: boolean;      // 선택된 아이템
};

export default function LLMChatScreen({
  onBack,
  onNavigate,
}: {
  onBack: () => void;
  onNavigate: (step: MainScreen) => void;
}) {
  const [userId, setUserId] = useState<number | null>(null);
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatRecommendations, setChatRecommendations] = useState<WardrobeItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [selectedItemIds, setSelectedItemIds] = useState<number[]>([]);

  // 사용자 ID 불러오기
  useEffect(() => {
    const loadUserId = async () => {
      console.log('💾 사용자 정보 로딩 시작...');
      try {
        const userData = await AsyncStorage.getItem('@kko/user');
        console.log('📦 AsyncStorage 데이터:', userData);
        if (userData) {
          const user = JSON.parse(userData);
          console.log('👤 파싱된 사용자 정보:', user);
          console.log('🆔 사용자 ID:', user.id || user.user_id);
          setUserId(user.id || user.user_id);
        } else {
          console.log('⚠️ AsyncStorage에 사용자 정보 없음');
        }
      } catch (error) {
        console.error('❌ 사용자 ID 로드 실패:', error);
      }
    };
    loadUserId();
  }, []);

  // 옷장 데이터 불러오기
  const fetchWardrobe = useCallback(async () => {
    console.log('👕 옷장 데이터 로딩 시작... userId:', userId);
    if (!userId) {
      console.log('⚠️ userId 없음 - 옷장 데이터 로드 취소');
      return;
    }
    
    try {
      const url = `${API_BASE_URL}/api/wardrobe/${userId}?include_defaults=false`;
      console.log('📡 API 호출:', url);
      const response = await fetch(url);
      console.log('📥 응답 상태:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ 옷장 데이터 로드 성공:', data.items.length, '개');
        
        // 중복 제거
        const uniqueItems = data.items.filter((item: WardrobeItem, index: number, self: WardrobeItem[]) => 
          index === self.findIndex((t: WardrobeItem) => t.id === item.id)
        );
        
        console.log('🔄 중복 제거 후:', uniqueItems.length, '개');
        setWardrobeItems(uniqueItems);
      } else {
        console.error('❌ 옷장 데이터 로드 실패:', response.status);
      }
    } catch (error) {
      console.error('❌ 옷장 데이터 로드 실패:', error);
    }
  }, [userId]);

  useEffect(() => {
    if (userId) {
      fetchWardrobe();
    }
  }, [userId, fetchWardrobe]);

  // 아이템 선택 (단일 선택만 가능)
  const toggleItemSelection = (itemId: number) => {
    setSelectedItemIds(prev => {
      // 이미 선택된 아이템을 다시 클릭하면 선택 해제
      if (prev.includes(itemId)) {
        return [];
      } else {
        // 새 아이템 선택 (기존 선택은 자동 해제)
        return [itemId];
      }
    });
  };

  // LLM 채팅 요청
  const sendChatMessage = async () => {
    console.log('\n🚀 sendChatMessage 호출됨!');
    console.log('📝 입력값:', chatInput);
    console.log('👤 userId:', userId);
    console.log('👕 선택된 아이템:', selectedItemIds);
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

    console.log('✅ 사용자 메시지 생성:', userMessage.content);
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      // FormData로 전송 (백엔드 요구사항)
      const formData = new FormData();
      formData.append('user_id', userId.toString());
      formData.append('message', userMessage.content);
      
      // 선택된 아이템 ID 추가
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

      if (response.ok) {
        const data = await response.json();
        console.log('📦 전체 응답 데이터:', data);
        console.log('🎯 추천 아이템 수:', data.recommendations?.length || 0);
        console.log('🎯 추천 아이템 샘플:', data.recommendations?.[0]);
        
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        };

        setChatMessages(prev => [...prev, assistantMessage]);
        
        // 추천 아이템이 있으면 표시
        if (data.recommendations && data.recommendations.length > 0) {
          console.log('✅ 추천 아이템 설정:', data.recommendations.length, '개');
          
          // 백엔드 형식을 프론트엔드 형식으로 변환
          const formattedRecommendations = data.recommendations.map((rec: any) => {
            // 아이템 이름 생성
            let itemName = '';
            const categories = [];
            if (rec.has_dress) categories.push('원피스');
            if (rec.has_outer) categories.push('아우터');
            if (rec.has_top) categories.push('상의');
            if (rec.has_bottom) categories.push('하의');
            
            itemName = categories.length > 0 ? categories.join(' / ') : `아이템 ${rec.item_id || rec.id}`;
            
            return {
              id: rec.item_id || rec.id,
              name: rec.name || itemName,
              brand: rec.is_default ? '기본 아이템' : 'My Wardrobe',
              category: rec.has_top ? 'top' : rec.has_bottom ? 'bottom' : rec.has_outer ? 'outer' : rec.has_dress ? 'dress' : 'other',
              color: '',
              fit: '',
              materials: [],
              image: rec.image_path || '',
              image_path: rec.image_path || '',
              has_top: rec.has_top,
              has_bottom: rec.has_bottom,
              has_outer: rec.has_outer,
              has_dress: rec.has_dress,
              is_recommended: true,  // 👈 추천 아이템 표시용
            };
          });
          
          // 선택된 아이템 정보도 함께 표시 (추천 결과 앞에 배치)
          // 주의: selectedItemIds를 초기화하기 전에 필터링해야 함
          const currentlySelected = [...selectedItemIds];  // 복사본 생성
          
          const selectedItems = chatRecommendations.filter(item => 
            currentlySelected.includes(item.id)
          ).map(item => ({
            ...item,
            is_selected: true  // 👈 선택된 아이템 표시용
          }));
          
          console.log('🎨 변환된 추천 아이템:', formattedRecommendations);
          console.log('👕 선택된 아이템:', selectedItems);
          console.log('🔄 선택 초기화 전 selectedItemIds:', selectedItemIds);
          
          // 선택 초기화 (먼저 실행)
          setSelectedItemIds([]);
          console.log('✅ 선택 초기화 완료');
          
          // 선택된 아이템 + 추천 아이템 함께 표시
          setChatRecommendations([...selectedItems, ...formattedRecommendations]);
        } else {
          console.log('⚠️ 추천 아이템이 없음');
        }
      } else {
        throw new Error(`HTTP ${response.status}`);
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

  // 초기 인사 메시지
  useEffect(() => {
    console.log('💬 초기 메시지 체크...');
    console.log('  - 옷장 아이템 수:', wardrobeItems.length);
    console.log('  - 채팅 메시지 수:', chatMessages.length);
    
    if (chatMessages.length === 0 && userId) {
      console.log('✅ 초기 인사 메시지 생성');
      const welcomeMessage: ChatMessage = {
        role: 'assistant',
        content: wardrobeItems.length > 0 
          ? `안녕하세요! 저는 당신의 패션 스타일리스트 AI입니다. 옷장에 ${wardrobeItems.length}개의 아이템이 있네요. 어떤 스타일링을 도와드릴까요?`
          : `안녕하세요! 저는 당신의 패션 스타일리스트 AI입니다. 어떤 스타일링을 도와드릴까요?`,
        timestamp: new Date(),
      };
      setChatMessages([welcomeMessage]);
    }
  }, [wardrobeItems, chatMessages.length, userId]);

  // 권한 요청
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

  // 이미지 업로드
  const uploadImage = async (imageUri: string) => {
    if (!userId) {
      Alert.alert('오류', '사용자 정보를 불러올 수 없습니다.');
      return;
    }

    setUploading(true);
    setChatLoading(true);

    const uploadingMessage: ChatMessage = {
      role: 'assistant',
      content: '📸 사진 분석 중입니다...',
      timestamp: new Date(),
    };
    setChatMessages(prev => [...prev, uploadingMessage]);

    try {
      const formData = new FormData();
      
      if (Platform.OS === 'web') {
        const response = await fetch(imageUri);
        const blob = await response.blob();
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
          'Accept': 'application/json',
        },
      });

      const data = await uploadResponse.json();
      console.log('📦 업로드 응답:', data);

      // 업로드 중 메시지 제거
      setChatMessages(prev => prev.filter(msg => msg.content !== '📸 사진 분석 중입니다...'));

      if (data.success) {
        // AI 응답 메시지
        const aiMessage: ChatMessage = {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
        };
        setChatMessages(prev => [...prev, aiMessage]);

        // 업로드된 아이템 카드로 표시
        if (data.uploaded_item) {
          setChatRecommendations([data.uploaded_item]);
        }

        // 옷장 갱신
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
      
      // 업로드 중 메시지 제거
      setChatMessages(prev => prev.filter(msg => msg.content !== '📸 사진 분석 중입니다...'));
      
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

  // 이미지 선택 옵션
  const showImageOptions = () => {
    if (uploading) return;

    if (Platform.OS === 'web') {
      pickImage();
    } else {
      Alert.alert(
        '사진 추가',
        '어떻게 추가하시겠어요?',
        [
          { text: '📸 카메라로 촬영', onPress: takePhoto },
          { text: '🖼️ 갤러리에서 선택', onPress: pickImage },
          { text: '취소', style: 'cancel' },
        ]
      );
    }
  };

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
              <Text style={[
                styles.messageText,
                message.role === 'user' ? styles.userMessageText : styles.assistantMessageText,
              ]}>
                {message.content}
              </Text>
              <Text style={styles.messageTime}>
                {message.timestamp.toLocaleTimeString('ko-KR', { 
                  hour: '2-digit', 
                  minute: '2-digit' 
                })}
              </Text>
            </View>
          ))}
          
          {chatLoading && (
            <View style={[styles.messageContainer, styles.assistantMessage]}>
              <ActivityIndicator size="small" color="#6B7280" />
              <Text style={[styles.messageText, styles.assistantMessageText, { marginLeft: 8 }]}>
                AI가 답변을 준비 중입니다...
              </Text>
            </View>
          )}
        </ScrollView>

        {/* 선택한 옷 섹션 */}
        {chatRecommendations.some(item => item.is_selected) && (
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
                      onPress={() => {}}  // 클릭 불가
                    >
                      <Image 
                        source={{ uri: `${API_BASE_URL}${item.image_path || item.image}` }} 
                        style={styles.recommendationImage}
                        onError={(e) => console.error('❌ 이미지 로드 실패:', `${API_BASE_URL}${item.image_path || item.image}`, e.nativeEvent.error)}
                        onLoad={() => console.log('✅ 이미지 로드 성공:', `${API_BASE_URL}${item.image_path || item.image}`)}
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

        {/* 추천 코디 섹션 */}
        {chatRecommendations.some(item => item.is_recommended) && (
          <View style={styles.recommendationsContainer}>
            <View style={styles.recommendationsHeader}>
              <Text style={styles.recommendationsTitle}>✨ 추천 코디</Text>
              {!chatRecommendations.some(item => item.is_selected) && (
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
                    // 실시간 선택 상태 (체크 마크용)
                    const isCurrentlySelected = selectedItemIds.includes(item.id);
                    
                    return (
                      <Pressable 
                        key={`recommended-${item.id}-${index}`} 
                        style={[
                          styles.recommendationCard,
                          isCurrentlySelected && styles.recommendationCardSelected
                        ]}
                        onPress={() => toggleItemSelection(item.id)}
                      >
                        <Image 
                          source={{ uri: `${API_BASE_URL}${item.image_path || item.image}` }} 
                          style={styles.recommendationImage}
                          onError={(e) => console.error('❌ 이미지 로드 실패:', `${API_BASE_URL}${item.image_path || item.image}`, e.nativeEvent.error)}
                          onLoad={() => console.log('✅ 이미지 로드 성공:', `${API_BASE_URL}${item.image_path || item.image}`)}
                        />
                        <View style={styles.recommendedBadge}>
                          <Text style={styles.recommendedBadgeText}>추천</Text>
                        </View>
                        {/* 현재 선택 중인 아이템 체크 마크 (실시간) */}
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

        {/* 옷장 전체 보기 (선택도 추천도 아닌 경우) */}
        {chatRecommendations.length > 0 && 
         !chatRecommendations.some(item => item.is_selected || item.is_recommended) && (
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
                        isCurrentlySelected && styles.recommendationCardSelected
                      ]}
                      onPress={() => toggleItemSelection(item.id)}
                    >
                      <Image 
                        source={{ uri: `${API_BASE_URL}${item.image_path || item.image}` }} 
                        style={styles.recommendationImage}
                        onError={(e) => console.error('❌ 이미지 로드 실패:', `${API_BASE_URL}${item.image_path || item.image}`, e.nativeEvent.error)}
                        onLoad={() => console.log('✅ 이미지 로드 성공:', `${API_BASE_URL}${item.image_path || item.image}`)}
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
            style={[styles.sendButton, (!chatInput.trim() || chatLoading || uploading) && styles.sendButtonDisabled]}
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
    marginBottom: 16,  // 섹션 간 간격 증가
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
    marginBottom: BOTTOM_NAV_HEIGHT + 8, // 네비게이션 바 위에 배치
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
    flexShrink: 0, // 버튼이 줄어들지 않도록
    elevation: 2, // 안드로이드 그림자
    shadowColor: '#000', // iOS 그림자
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  sendButtonDisabled: {
    backgroundColor: '#D1D5DB',
  },
});
