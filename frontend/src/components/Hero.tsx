'use client';
import { useEffect, useRef } from 'react';
import Link from 'next/link';
import gsap from 'gsap';

export default function Hero() {
    const heroRef = useRef(null);

    useEffect(() => {
        const ctx = gsap.context(() => {
            // Entrance animations
            gsap.from(".hero-title", {
                opacity: 0,
                y: -80,
                duration: 1.2,
                ease: "power3.out",
                delay: 0.3
            });
            gsap.from(".hero-subtitle", {
                opacity: 0,
                y: 50,
                duration: 1,
                ease: "power2.out",
                delay: 0.6
            });
            gsap.from(".hero-buttons", {
                opacity: 0,
                scale: 0.9,
                duration: 0.8,
                ease: "back.out(1.7)",
                delay: 0.9
            });
            gsap.from(".hero-stats", {
                opacity: 0,
                y: 30,
                duration: 1,
                stagger: 0.2,
                delay: 1.2
            });
        }, heroRef);
        return () => ctx.revert();
    }, []);

    return (
        <section className="hero-modern" ref={heroRef}>
            {/* Animated gradient background */}
            <div className="hero-gradient-bg"></div>

            {/* Floating shapes */}
            <div className="floating-shapes">
                <div className="shape shape-1"></div>
                <div className="shape shape-2"></div>
                <div className="shape shape-3"></div>
            </div>

            <div className="container" style={{ position: 'relative', zIndex: 2, paddingTop: '120px', paddingBottom: '80px' }}>
                <div className="row align-items-center">
                    <div className="col-lg-7">
                        <div className="hero-content">
                            {/* Badge */}
                            <div className="hero-badge mb-4">
                                <span className="badge-glow">
                                    <i className="ph ph-sparkle me-2"></i>
                                    AI-Powered Fitness
                                </span>
                            </div>

                            {/* Main Title */}
                            <h1 className="hero-title" style={{
                                fontSize: 'clamp(2.5rem, 8vw, 5rem)',
                                fontWeight: '900',
                                lineHeight: '1.1',
                                marginBottom: '1.5rem',
                                background: 'linear-gradient(135deg, #fff 0%, #00ccff 50%, #8b5cf6 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text'
                            }}>
                                Transform Your Body<br />
                                <span style={{
                                    background: 'linear-gradient(135deg, #00ccff 0%, #0077ff 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent'
                                }}>
                                    With AI
                                </span>
                            </h1>

                            {/* Subtitle */}
                            <p className="hero-subtitle" style={{
                                fontSize: '1.3rem',
                                color: 'rgba(255, 255, 255, 0.85)',
                                marginBottom: '2.5rem',
                                maxWidth: '600px',
                                lineHeight: '1.6'
                            }}>
                                Get personalized 30-day workout and nutrition plans powered by cutting-edge machine learning. Your journey to peak fitness starts here.
                            </p>

                            {/* CTA Buttons */}
                            <div className="hero-buttons d-flex gap-3 flex-wrap">
                                <Link href="/register" className="btn-modern btn-primary-gradient">
                                    <i className="ph ph-rocket-launch me-2"></i>
                                    Start Free Trial
                                    <div className="btn-shine"></div>
                                </Link>
                                <Link href="/login" className="btn-modern btn-secondary-glass">
                                    <i className="ph ph-sign-in me-2"></i>
                                    Sign In
                                </Link>
                            </div>

                            {/* Stats */}
                            <div className="hero-stats d-flex gap-4 mt-5 flex-wrap">
                                <div className="stat-item">
                                    <div className="stat-number" style={{
                                        fontSize: '2.5rem',
                                        fontWeight: 'bold',
                                        background: 'linear-gradient(135deg, #00ccff, #0077ff)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent'
                                    }}>
                                        10K+
                                    </div>
                                    <div className="stat-label" style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>
                                        Active Users
                                    </div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-number" style={{
                                        fontSize: '2.5rem',
                                        fontWeight: 'bold',
                                        background: 'linear-gradient(135deg, #10b981, #059669)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent'
                                    }}>
                                        95%
                                    </div>
                                    <div className="stat-label" style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>
                                        Success Rate
                                    </div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-number" style={{
                                        fontSize: '2.5rem',
                                        fontWeight: 'bold',
                                        background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent'
                                    }}>
                                        30
                                    </div>
                                    <div className="stat-label" style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem' }}>
                                        Day Program
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right side - 3D Card */}
                    <div className="col-lg-5 d-none d-lg-block">
                        <div className="hero-3d-card">
                            <div className="card-glassmorphism" style={{
                                background: 'rgba(255, 255, 255, 0.05)',
                                backdropFilter: 'blur(20px)',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                borderRadius: '24px',
                                padding: '40px',
                                boxShadow: '0 20px 60px rgba(0, 204, 255, 0.3)',
                                transform: 'perspective(1000px) rotateY(-5deg)',
                                transition: 'transform 0.3s ease'
                            }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{
                                        fontSize: '5rem',
                                        marginBottom: '20px',
                                        filter: 'drop-shadow(0 0 30px rgba(0, 204, 255, 0.8))'
                                    }}>
                                        🏋️
                                    </div>
                                    <h3 style={{ color: '#00ccff', marginBottom: '15px', fontSize: '1.5rem' }}>
                                        Smart AI Training
                                    </h3>
                                    <p style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                        Our advanced machine learning algorithms create personalized plans that adapt to your unique body and goals.
                                    </p>

                                    {/* Progress indicator */}
                                    <div className="mt-4">
                                        <div style={{
                                            background: 'rgba(0, 204, 255, 0.1)',
                                            borderRadius: '12px',
                                            height: '8px',
                                            overflow: 'hidden'
                                        }}>
                                            <div style={{
                                                background: 'linear-gradient(90deg, #00ccff, #0077ff)',
                                                width: '75%',
                                                height: '100%',
                                                borderRadius: '12px',
                                                boxShadow: '0 0 20px rgba(0, 204, 255, 0.5)'
                                            }}></div>
                                        </div>
                                        <div className="d-flex justify-content-between mt-2">
                                            <span style={{ fontSize: '0.8rem', color: 'rgba(255, 255, 255, 0.5)' }}>Day 1</span>
                                            <span style={{ fontSize: '0.8rem', color: '#00ccff', fontWeight: 'bold' }}>Day 30</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Add custom styles */}
            <style jsx>{`
                .hero-modern {
                    min-height: 100vh;
                    position: relative;
                    overflow: hidden;
                    display: flex;
                    align-items: center;
                }

                .hero-gradient-bg {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 30%, #0f0f23 60%, #0a0a0a 100%);
                    z-index: 0;
                }

                .hero-gradient-bg::before {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(0, 204, 255, 0.1) 0%, transparent 70%);
                    animation: pulse 8s ease-in-out infinite;
                }

                @keyframes pulse {
                    0%, 100% { transform: translate(0, 0) scale(1); }
                    50% { transform: translate(10%, 10%) scale(1.1); }
                }

                .floating-shapes {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    top: 0;
                    left: 0;
                    z-index: 1;
                    overflow: hidden;
                }

                .shape {
                    position: absolute;
                    border-radius: 50%;
                    filter: blur(40px);
                    opacity: 0.3;
                    animation: float 15s infinite ease-in-out;
                }

                .shape-1 {
                    width: 400px;
                    height: 400px;
                    background: linear-gradient(135deg, #00ccff, #0077ff);
                    top: 10%;
                    left: 10%;
                    animation-delay: 0s;
                }

                .shape-2 {
                    width: 300px;
                    height: 300px;
                    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
                    top: 50%;
                    right: 10%;
                    animation-delay: 2s;
                }

                .shape-3 {
                    width: 250px;
                    height: 250px;
                    background: linear-gradient(135deg, #10b981, #059669);
                    bottom: 10%;
                    left: 30%;
                    animation-delay: 4s;
                }

                @keyframes float {
                    0%, 100% {
                        transform: translate(0, 0) rotate(0deg);
                    }
                    33% {
                        transform: translate(50px, -50px) rotate(120deg);
                    }
                    66% {
                        transform: translate(-50px, 50px) rotate(240deg);
                    }
                }

                .hero-badge .badge-glow {
                    display: inline-block;
                    padding: 8px 20px;
                    background: rgba(0, 204, 255, 0.1);
                    border: 1px solid rgba(0, 204, 255, 0.3);
                    border-radius: 50px;
                    color: #00ccff;
                    font-weight: 600;
                    font-size: 0.9rem;
                    box-shadow: 0 0 20px rgba(0, 204, 255, 0.3);
                    animation: glow 2s ease-in-out infinite;
                }

                @keyframes glow {
                    0%, 100% {
                        box-shadow: 0 0 20px rgba(0, 204, 255, 0.3);
                    }
                    50% {
                        box-shadow: 0 0 30px rgba(0, 204, 255, 0.6);
                    }
                }

                .btn-modern {
                    position: relative;
                    padding: 16px 36px;
                    font-size: 1.1rem;
                    font-weight: 700;
                    border-radius: 12px;
                    border: none;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    overflow: hidden;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                }

                .btn-primary-gradient {
                    background: linear-gradient(135deg, #00ccff, #0077ff);
                    color: white;
                    box-shadow: 0 10px 30px rgba(0, 204, 255, 0.4);
                }

                .btn-primary-gradient:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 15px 40px rgba(0, 204, 255, 0.6);
                }

                .btn-secondary-glass {
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    color: white;
                }

                .btn-secondary-glass:hover {
                    background: rgba(255, 255, 255, 0.1);
                    transform: translateY(-3px);
                }

                .btn-shine {
                    position: absolute;
                    top: -50%;
                    left: -100%;
                    width: 50%;
                    height: 200%;
                    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                    animation: shine 3s infinite;
                }

                @keyframes shine {
                    0% {
                        left: -100%;
                    }
                    100% {
                        left: 200%;
                    }
                }

                .hero-3d-card:hover .card-glassmorphism {
                    transform: perspective(1000px) rotateY(0deg) scale(1.02);
                }
            `}</style>
        </section>
    );
}
