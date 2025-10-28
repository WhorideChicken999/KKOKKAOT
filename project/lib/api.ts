// PROJECT01/lib/api.ts
import axios from 'axios';

// 🚨 1단계에서 찾은 내 PC의 IP 주소를 여기에 입력하세요!
const API_URL = 'http://192.168.0.97:4000';

const apiClient = axios.create({
  baseURL: API_URL,
});

// '/api/users' 주소로 데이터를 요청하는 함수
export const getUsers = () => {
  return apiClient.get('/api/users');
};