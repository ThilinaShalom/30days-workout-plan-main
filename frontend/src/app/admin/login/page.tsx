'use client';
import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/Navbar';
import api from '@/lib/api';

export default function AdminLogin() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        try {
            const response = await api.post('/admin/login', { username, password });
            if (response.data.success) {
                router.push('/admin/dashboard');
            }
        } catch (err: any) {
            setError(err.response?.data?.error || 'Login failed');
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
                    maxWidth: '450px',
                    padding: 'clamp(20px, 5vw, 40px)'
                }}>
                    <div className="text-center mb-4">
                        <i className="ph ph-shield-check" style={{
                            fontSize: 'clamp(2.5rem, 8vw, 3rem)',
                            color: '#00ccff'
                        }}></i>
                        <h2 className="mt-3" style={{
                            fontSize: 'clamp(1.5rem, 5vw, 2rem)'
                        }}>
                            Admin Login
                        </h2>
                    </div>
                    {error && <div className="alert alert-danger">{error}</div>}
                    <form onSubmit={handleSubmit}>
                        <div className="form-group mb-3">
                            <input
                                type="text"
                                className="form-control"
                                placeholder="Admin Username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                            />
                        </div>
                        <div className="form-group mb-3" style={{ position: 'relative' }}>
                            <input
                                type={showPassword ? 'text' : 'password'}
                                className="form-control"
                                placeholder="Admin Password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
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
                                    padding: '4px 8px'
                                }}
                            >
                                <i className={`ph ${showPassword ? 'ph-eye-slash' : 'ph-eye'}`}></i>
                            </button>
                        </div>
                        <button type="submit" className="btn btn-primary w-100">
                            <i className="ph ph-unlock me-2"></i>
                            Admin Login
                        </button>
                    </form>
                    <div className="text-center mt-3">
                        <p style={{ fontSize: '0.85rem', color: '#888' }}>
                            <i className="ph ph-info me-1"></i>
                            Credentials: admin / admin123
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}
