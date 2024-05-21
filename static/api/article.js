import request from '@/utils/request'

// 获取文章列表
// export const getArticleList = ({ current, pageSize }) => {
//   return request.get('/admin/interview/query', {
//     params: {
//       current,
//       pageSize
//     }
//   })
// }

// 新增文章
export const createArticle = obj => {
  return request.post('/admin/interview/create', obj)
}

// 删除文章
export const removeArticle = (id) => {
  return request.delete('/admin/interview/remove', {
    // 参数传递在请求体
    data: {
      id
    }
  })
}

// 获取单个面经详情
export const getArticleDetail = id => {
  return request.get('/admin/interview/show', {
    params: {
      id
    }
  })
}

// 用户修改了内容, 一点击提交按钮, 进行更新操作
export const updateArticle = data => {
  return request.put('/admin/interview/update', data)
}

// 对于get和delete, 如果要传递参数, 必须先传对象, 对象里面配置 params(地址栏) 或 data(请求体)
// axios.get(url, config)
// axios.delete(url, config)

// 对于下面三种情况, 第二个直接写参数对象即可
// axios.put(url, data, config)
// axios.patch(url, data, config)
// axios.post(url, data, config)
