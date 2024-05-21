import request from '@/utils/request'

// export const login = (obj) => {
//   return request.post('/auth/login', {
//     username: obj.username,
//     password: obj.password
//   })
// }

// 登录api的封装
// 从传过来的对象中，解构出 username 和 password
export const login = ({ username, password }) => {
  return request.post('/user/login', {
    username,
    password
  })
}

export const uploadText = (formData) => {
  const path = '/check/text'
  // 然后我们用formData作为POST请求的body
  return request.post(path, formData, {})
}

export const uploadImage = (file) => {
  // 这里我们创建了一个FormData对象并将文件附加到它
  const path = '/check/image'
  // 然后我们用formData作为POST请求的body
  return request.post(path, file, {})
}

export const uploadOCRimage = (file) => {
  // 这里我们创建了一个FormData对象并将文件附加到它
  const path = '/check/OCR'
  // 然后我们用formData作为POST请求的body
  return request.post(path, file, {})
}
// 获取当前的用户信息
export const getUser = () => {
  return request.get('/user/info')
}
// 更新当前用户信息
export const updateUser = (userData) => {
  return request.post('/user/update', userData)
}

export const register = ({ username, password }) => {
  return request.post('/user/register', {
    username,
    password
  })
}

export const checkUsernameExist = (username) => {
  return request.get(`/user/check-username/${username}`)
}
