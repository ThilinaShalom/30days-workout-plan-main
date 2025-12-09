'use client';
import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/Navbar';
import api from '@/lib/api';

export default function Register() {
    const [formData, setFormData] = useState({
        user_name: '',
        email: '',
        password: '',
        user_type: 'customer'
    });
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const router = useRouter();

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        try {
            const response = await api.post('/auth/register', formData);
            if (response.data.success) {
                router.push('/login');
            }
        } catch (err: any) {
            setError(err.response?.data?.error || 'Registration failed');
        }
    };

    return (
        <>
            <Navbar />
            <div className="container" style={{
                minHeight: '100vh',
                marginTop: '60px',
                padding: '20px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
            }}>
                <div className="auth-container" style={{
                    width: '100%',
                    maxWidth: '500px',
                    padding: 'clamp(20px, 5vw, 40px)'
                }}>
                    <div className="text-center mb-4">
                        <i className="ph ph-user-plus" style={{
                            fontSize: 'clamp(2.5rem, 8vw, 3rem)',
                            color: '#00ccff'
                        }}></i>
                        <h2 className="mt-3" style={{
                            fontSize: 'clamp(1.5rem, 5vw, 2rem)'
                        }}>
                            Create Account
                        </h2>
                    </div>
                    {error && <div className="alert alert-danger">{error}</div>}
                    <form onSubmit={handleSubmit}>
                        <div className="form-group mb-3">
                            <div className="input-group">
                                <span className="input-group-text" style={{ background: 'rgba(0, 204, 255, 0.1)', borderColor: 'rgba(0, 204, 255, 0.3)', color: '#00ccff' }}>
                                    <i className="ph ph-user"></i>
                                </span>
                                <input
                                    type="text"
                                    className="form-control"
                                    name="user_name"
                                    placeholder="Enter your name"
                                    value={formData.user_name}
                                    onChange={handleChange}
                                    required
                                />
                            </div>
                        </div>

                        <div className="form-group mb-3">
                            <div className="input-group">
                                <span className="input-group-text" style={{ background: 'rgba(0, 204, 255, 0.1)', borderColor: 'rgba(0, 204, 255, 0.3)', color: '#00ccff' }}>
                                    <i className="ph ph-envelope"></i>
                                </span>
                                <input
                                    type="email"
                                    className="form-control"
                                    name="email"
                                    placeholder="Enter your email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required
                                />
                            </div>
                        </div>

                        <div className="form-group mb-3">
                            <div className="input-group" style={{ position: 'relative' }}>
                                <span className="input-group-text" style={{ background: 'rgba(0, 204, 255, 0.1)', borderColor: 'rgba(0, 204, 255, 0.3)', color: '#00ccff' }}>
                                    <i className="ph ph-lock"></i>
                                </span>
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    className="form-control"
                                    name="password"
                                    placeholder="Create password (min 6 characters)"
                                    value={formData.password}
                                    onChange={handleChange}
                                    required
                                    minLength={6}
                                    style={{ paddingRight: '45px' }}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    style={{
                                        position: 'absolute',
                                        right: '12px',
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        background: 'transparent',
                                        border: 'none',
                                        color: '#00ccff',
                                        cursor: 'pointer',
                                        fontSize: '1.2rem',
                                        padding: '4px 8px',
                                        zIndex: 10
                                    }}
                                >
                                    <i className={`ph ${showPassword ? 'ph-eye-slash' : 'ph-eye'}`}></i>
                                </button>
                            </div>
                        </div>

                        <div className="form-group mb-4">
                            <div className="input-group">
                                <span className="input-group-text" style={{ background: 'rgba(0, 204, 255, 0.1)', borderColor: 'rgba(0, 204, 255, 0.3)', color: '#00ccff' }}>
                                    <i className="ph ph-users"></i>
                                </span>
                                <select
                                    className="form-select"
                                    name="user_type"
                                    value={formData.user_type}
                                    onChange={handleChange}
                                    required
                                    style={{ background: 'rgba(255, 255, 255, 0.1)', color: '#fff' }}
                                >
                                    <option value="customer">Customer</option>
                                </select>
                            </div>
                        </div>

                        <button type="submit" className="btn btn-primary w-100 mb-3">
                            <i className="ph ph-user-plus me-2"></i>Create Account
                        </button>
                    </form>
                    <div className="text-center">
                        <p style={{ color: '#ccc' }}>Already have an account?
                            <Link href="/login" style={{ color: '#00ccff' }} className="ms-1">Login here</Link>
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}
