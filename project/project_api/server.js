// server.js

// 1. 필요한 라이브러리들을 불러옵니다.
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const bcrypt = require('bcrypt'); // [추가] 비밀번호 암호화를 위해 bcrypt 추가

// 2. Express 앱을 생성하고 기본 설정을 합니다.
const app = express();
const port = 4000; // 서버를 실행할 포트 번호

app.use(cors()); // 다른 주소(React Native 앱)에서의 요청을 허용
app.use(express.json()); // JSON 형태의 데이터를 서버가 이해할 수 있도록 설정

// 3. 🛡️ PostgreSQL 데이터베이스 연결 설정을 합니다.
const pool = new Pool({
  user: 'postgres',         // 👈 스크린샷에서 확인한 정확한 사용자 이름
  host: 'localhost',
  database: 'postgres',     // 👈 스크린샷에서 확인한 정확한 데이터베이스 이름
  password: '1234', // 👈 이 부분은 직접 수정해주세요!
  port: 5432,
});
// 4. ⚙️ API 주소(엔드포인트)를 만듭니다.

// --- 기존 API (테스트 및 사용자 목록 조회) ---
app.get('/', (req, res) => {
  res.send('✅ API 서버 정상 작동');
});

app.get('/api/users', async (req, res) => {
  try {
    const result = await pool.query('SELECT id, name, email, age_group, created_at FROM users'); // 비밀번호는 제외하고 조회
    res.json(result.rows);
  } catch (error) {
    console.error('DB 조회 에러:', error);
    res.status(500).json({ error: '데이터베이스 처리 중 오류가 발생했습니다.' });
  }
});


app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // 1. 이메일과 비밀번호가 모두 입력되었는지 확인
    if (!email || !password) {
      return res.status(400).json({ success: false, message: '이메일과 비밀번호를 모두 입력해주세요.' });
    }

    // 2. DB에서 이메일로 사용자 찾기
    const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
    const user = result.rows[0];

    if (!user) {
      // 사용자가 존재하지 않는 경우
      return res.status(401).json({ success: false, message: '존재하지 않는 이메일입니다.' });
    }

    // 3. DB에 저장된 암호화된 비밀번호와 사용자가 입력한 비밀번호 비교
    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      // 비밀번호가 일치하지 않는 경우
      return res.status(401).json({ success: false, message: '비밀번호가 일치하지 않습니다.' });
    }

    // 4. 로그인 성공: DB에서 가져온 사용자 정보 반환 (비밀번호 제외)
    res.status(200).json({
      success: true,
      message: '로그인에 성공했습니다.',
      user: {
        user_id: user.id,  // ← user_id로 수정
        name: user.name, // DB에 저장된 실제 이름
        email: user.email,
      },
    });

  } catch (error) {
    console.error('로그인 처리 중 오류 발생:', error);
    res.status(500).json({ success: false, message: '서버 내부 오류가 발생했습니다.' });
  }
});



// --- [추가] 회원가입 API 엔드포인트 ---
app.post('/api/signup', async (req, res) => {
  try {
    // 1. 프론트엔드로부터 받은 데이터 추출
    const { name, email, password, ageGroup, stylePreferences } = req.body;

    // 2. 데이터 유효성 검사
    if (!name || !email || !password || !ageGroup) {
      return res.status(400).json({ success: false, message: '필수 정보를 모두 입력해주세요.' });
    }

    // 3. DB에서 이메일 중복 확인
    const existingUser = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
    if (existingUser.rows.length > 0) {
      return res.status(409).json({ success: false, message: '이미 사용 중인 이메일입니다.' });
    }

    // 4. 비밀번호 암호화
    const hashedPassword = await bcrypt.hash(password, 10);

    // 5. DB에 새로운 사용자 정보 저장
    // stylePreferences는 배열이므로 JSON 형태로 변환하여 저장합니다.
    const newUserQuery = `
      INSERT INTO users (name, email, password, age_group, style_preferences)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id, name, email
    `;
    const values = [name, email, hashedPassword, ageGroup, JSON.stringify(stylePreferences)];
    
    const result = await pool.query(newUserQuery, values);
    const newUser = result.rows[0];

    // 6. 성공 응답 전송
    res.status(201).json({
      success: true,
      message: '회원가입이 완료되었습니다.',
      user: newUser,
    });

  } catch (error) {
    console.error('회원가입 처리 중 오류 발생:', error);
    res.status(500).json({ success: false, message: '서버 내부 오류가 발생했습니다.' });
  }
});


// 5. 설정한 포트 번호로 서버 실행을 시작합니다.
app.listen(port, () => {
  console.log(`🚀 백엔드 API 서버가 http://localhost:${port} 에서 실행 중입니다.`);
});