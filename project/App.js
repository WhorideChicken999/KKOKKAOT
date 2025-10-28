// // App.js
// import { useCallback, useMemo, useState } from 'react';
// import { SafeAreaProvider } from 'react-native-safe-area-context';
// import { useAppFonts } from './lib/useAppFonts';

// // 화면들
// import AuthScreen from './components/figma/AuthScreen';
// import BodyPhotoSetup from './components/figma/BodyPhotoSetup';
// import DailyOutfitRecommendation from './components/figma/DailyOutfitRecommendation';
// import HomeScreen from './components/figma/homescreen';
// import ShoppingRecommendations from './components/figma/ShoppingRecommendations';
// import SplashScreen from './components/figma/SplashScreen';
// import StyleAnalysisDetail from './components/figma/StyleAnalysisDetail';
// import UserProfileSetup from './components/figma/UserProfileSetup';
// import WardrobeManagement from './components/figma/WardrobeManagement';
// import WardrobeSetup from './components/figma/WardrobeSetup';

// // 임시
// import PlaceholderScreen from './screens/PlaceholderScreen';

// export default function App() {
//   const fontsLoaded = useAppFonts();

//   // 온보딩 흐름
//   const [step, setStep] = useState('splash');

//   // 회원 이름(회원가입 시 저장 → 홈 인사말에 표시)
//   const [userName, setUserName] = useState('사용자');

//   const titles = useMemo(
//     () => ({
//       home: '홈',
//       'today-curation': '오늘의 큐레이션',
//       'daily-outfit': '오늘 뭐 입지?',
//       'wardrobe-management': '내 옷장',
//       'style-analysis': '스타일 분석',
//       shopping: '쇼핑 추천',
//       'recent-styling': '최근 스타일링',
//       'blocked-outfits': '차단 코디',
//       'virtual-fitting': '나만의 가상 피팅',
//       // 온보딩
//       auth: '로그인/회원가입',
//       'user-profile-setup': '사용자 정보 설정',
//       'body-photo-setup': '체형 사진 업로드',
//       'wardrobe-setup': '가상 옷장 설정',
//     }),
//     []
//   );

//   // 회원가입 단계 완료 시 이름 받아서 저장
//   const handleProfileComplete = useCallback((payload) => {
//     // UserProfileSetup에서 onComplete({ name }) 처럼 넘겨주면 반영.
//     if (payload && typeof payload.name === 'string' && payload.name.trim()) {
//       setUserName(payload.name.trim());
//     }
//     setStep('body-photo-setup');
//   }, []);

//   if (!fontsLoaded) return null;

//   return (
//     <SafeAreaProvider>
//       {step === 'splash' ? (
//         <SplashScreen onGetStarted={() => setStep('auth')} />
//       ) : step === 'auth' ? (
//         // AuthScreen은 아래 3개 중 **아무거나** 호출해도 되도록 백워드 호환:
//         // - onLogin(): 로그인 성공시 → 바로 홈
//         // - onSignup(): 회원가입 플로우 시작 → 프로필 설정
//         // - onAuth(): (구버전) 기본은 회원가입 플로우로 간주
//         <AuthScreen
//           onLogin={() => setStep('home')}
//           onSignup={() => setStep('user-profile-setup')}
//           onAuth={() => setStep('user-profile-setup')}
//         />
//       ) : step === 'user-profile-setup' ? (
//         // onComplete에서 { name } 넘겨주면 App이 인사말에 사용
//         <UserProfileSetup onComplete={(data) => handleProfileComplete(data)} />
//       ) : step === 'body-photo-setup' ? (
//         <BodyPhotoSetup onComplete={() => setStep('wardrobe-setup')} />
//       ) : step === 'wardrobe-setup' ? (
//         <WardrobeSetup onComplete={() => setStep('home')} onSkip={() => setStep('home')} />
//       ) : step === 'wardrobe-management' ? (
//         <WardrobeManagement onBack={() => setStep('home')} />
//       ) : step === 'daily-outfit' ? (
//         <DailyOutfitRecommendation onBack={() => setStep('home')} />
//       ) : step === 'style-analysis' ? (
//         <StyleAnalysisDetail onBack={() => setStep('home')} />
//       ) : step === 'shopping' ? (
//         <ShoppingRecommendations onBack={() => setStep('home')} />
//       ) : step === 'home' ? (
//         <HomeScreen
//           onNavigate={(s) => setStep(s)}
//           userName={userName}
//           lovedOutfitsCount={3}
//           blockedOutfitsCount={1}
//         />
//       ) : (
//         <PlaceholderScreen
//           title={titles[step] || String(step)}
//           onBack={() => {
//             if (step === 'wardrobe-setup') setStep('body-photo-setup');
//             else if (step === 'user-profile-setup' || step === 'body-photo-setup') setStep('auth');
//             else setStep('home');
//           }}
//         />
//       )}
//     </SafeAreaProvider>
//   );
// }
