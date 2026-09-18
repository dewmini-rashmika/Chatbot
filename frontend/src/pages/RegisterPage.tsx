import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'
import { Music } from 'lucide-react'

export default function RegisterPage() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', username: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      await authAPI.register(form)
      navigate('/login')
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        setError(detail[0].msg);
      } else {
        setError(detail ?? 'Registration failed');
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 px-4">
      <div className="w-full max-w-md">
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-violet-600 rounded-xl flex items-center justify-center mb-3">
            <Music className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Create Account</h1>
          <p className="text-gray-400 text-sm mt-1">Start exploring music knowledge</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-gray-900 rounded-2xl p-8 border border-gray-800 space-y-5">
          {error && (
            <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">
              {error}
            </div>
          )}
          {(['email', 'username', 'password'] as const).map((field) => (
            <div key={field}>
              <label className="block text-sm font-medium text-gray-300 mb-2 capitalize">{field}</label>
              <input
                type={field === 'password' ? 'password' : field === 'email' ? 'email' : 'text'}
                required
                value={form[field]}
                onChange={(e) => setForm({ ...form, [field]: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                placeholder={field === 'email' ? 'you@example.com' : field === 'password' ? '8+ characters' : 'your_username'}
              />
            </div>
          ))}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white font-medium rounded-lg py-3 transition-colors"
          >
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
          <p className="text-center text-gray-400 text-sm">
            Already have an account?{' '}
            <Link to="/login" className="text-violet-400 hover:text-violet-300">Sign in</Link>
          </p>
        </form>
      </div>
    </div>
  )
}
