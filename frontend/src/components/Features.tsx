'use client';
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export default function Features() {
    const sectionRef = useRef(null);

    useEffect(() => {
        const ctx = gsap.context(() => {
            // Simple fade-in animation without opacity issues
            gsap.from(".features-title", {
                scrollTrigger: {
                    trigger: ".ai-features",
                    start: "top 80%",
                    toggleActions: "play none none none"
                },
                y: 30,
                duration: 0.8
            });

            gsap.from(".feature-card-modern", {
                scrollTrigger: {
                    trigger: ".ai-features",
                    start: "top 70%",
                    toggleActions: "play none none none"
                },
                y: 50,
                stagger: 0.15,
                duration: 0.6,
                ease: "power2.out"
            });
        }, sectionRef);
        return () => ctx.revert();
    }, []);

    const features = [
        {
            icon: "🧠",
            iconBg: "linear-gradient(135deg, #00ccff, #0077ff)",
            title: "Deep Learning Models",
            description: "Advanced neural networks analyze your fitness patterns and optimize your workout plans in real-time for maximum results.",
            color: "#00ccff"
        },
        {
            icon: "📊",
            iconBg: "linear-gradient(135deg, #10b981, #059669)",
            title: "Predictive Analytics",
            description: "Machine learning algorithms forecast your progress and dynamically adjust your training schedule for optimal performance.",
            color: "#10b981"
        },
        {
            icon: "⚡",
            iconBg: "linear-gradient(135deg, #f59e0b, #d97706)",
            title: "Smart Adaptation",
            description: "AI-powered system that evolves with you, learning from every workout to create increasingly effective training plans.",
            color: "#f59e0b"
        },
        {
            icon: "🎯",
            iconBg: "linear-gradient(135deg, #8b5cf6, #6d28d9)",
            title: "Goal Tracking",
            description: "Intelligent monitoring system that tracks your achievements and keeps you motivated throughout your fitness journey.",
            color: "#8b5cf6"
        },
        {
            icon: "🍎",
            iconBg: "linear-gradient(135deg, #ef4444, #dc2626)",
            title: "Nutrition Optimization",
            description: "Personalized meal plans calculated to perfection, balancing macros and calories for your specific body and goals.",
            color: "#ef4444"
        },
        {
            icon: "👥",
            iconBg: "linear-gradient(135deg, #06b6d4, #0891b2)",
            title: "Expert Coaching",
            description: "Get professional feedback from certified coaches who review and optimize your AI-generated fitness plans.",
            color: "#06b6d4"
        }
    ];

    return (
        <section className="ai-features py-5" ref={sectionRef} style={{
            background: 'linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%)',
            position: 'relative',
            overflow: 'hidden',
            paddingTop: '80px',
            paddingBottom: '80px'
        }}>
            <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '800px',
                height: '800px',
                background: 'radial-gradient(circle, rgba(0, 204, 255, 0.05) 0%, transparent 70%)',
                filter: 'blur(100px)',
                pointerEvents: 'none'
            }}></div>

            <div className="container" style={{ position: 'relative', zIndex: 2 }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <div style={{
                        display: 'inline-block',
                        padding: '8px 20px',
                        background: 'rgba(139, 92, 246, 0.2)',
                        border: '1px solid rgba(139, 92, 246, 0.5)',
                        borderRadius: '50px',
                        color: '#a78bfa',
                        fontWeight: '600',
                        fontSize: '0.9rem',
                        marginBottom: '1.5rem'
                    }}>
                        <i className="ph ph-magic-wand me-2"></i>
                        AI-Powered Features
                    </div>
                    <h2 className="features-title" style={{
                        fontSize: 'clamp(2rem, 5vw, 3.5rem)',
                        fontWeight: '900',
                        marginBottom: '1rem',
                        color: '#fff',
                        textShadow: '0 0 30px rgba(0, 204, 255, 0.5)'
                    }}>
                        Powered by Advanced AI
                    </h2>
                    <p style={{
                        fontSize: '1.2rem',
                        color: '#fff',
                        opacity: 0.8,
                        maxWidth: '700px',
                        margin: '0 auto'
                    }}>
                        Experience the future of fitness with machine learning technology that adapts to your unique needs
                    </p>
                </div>

                <div className="row g-4">
                    {features.map((feature, index) => (
                        <div className="col-lg-4 col-md-6" key={index}>
                            <div className="feature-card-modern" style={{
                                background: 'rgba(30, 30, 50, 0.95)',
                                backdropFilter: 'blur(20px)',
                                border: `3px solid ${feature.color}`,
                                borderRadius: '20px',
                                padding: '35px',
                                height: '100%',
                                transition: 'all 0.3s ease',
                                cursor: 'pointer',
                                position: 'relative',
                                overflow: 'hidden',
                                boxShadow: `0 8px 32px ${feature.color}40`
                            }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateY(-10px)';
                                    e.currentTarget.style.boxShadow = `0 20px 60px ${feature.color}60`;
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateY(0)';
                                    e.currentTarget.style.boxShadow = `0 8px 32px ${feature.color}40`;
                                }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    background: `linear-gradient(135deg, ${feature.color}10 0%, transparent 100%)`,
                                    opacity: 0,
                                    transition: 'opacity 0.3s ease',
                                    pointerEvents: 'none'
                                }} className="card-gradient-overlay"></div>

                                <div style={{
                                    width: '80px',
                                    height: '80px',
                                    background: feature.iconBg,
                                    borderRadius: '16px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '2.5rem',
                                    marginBottom: '1.5rem',
                                    boxShadow: `0 10px 30px ${feature.color}50`,
                                    position: 'relative',
                                    zIndex: 1
                                }}>
                                    {feature.icon}
                                </div>

                                <h3 style={{
                                    fontSize: '1.4rem',
                                    fontWeight: 'bold',
                                    color: '#fff',
                                    marginBottom: '1rem',
                                    position: 'relative',
                                    zIndex: 1,
                                    textShadow: '0 2px 10px rgba(0, 0, 0, 0.5)'
                                }}>
                                    {feature.title}
                                </h3>
                                <p style={{
                                    color: '#fff',
                                    opacity: 0.85,
                                    fontSize: '1rem',
                                    lineHeight: '1.7',
                                    marginBottom: 0,
                                    position: 'relative',
                                    zIndex: 1
                                }}>
                                    {feature.description}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>

                <div style={{ textAlign: 'center', marginTop: '4rem' }}>
                    <p style={{ color: '#fff', opacity: 0.7, fontSize: '1.1rem', marginBottom: '1.5rem' }}>
                        Ready to experience the power of AI-driven fitness?
                    </p>
                    <a href="/register" style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        padding: '14px 32px',
                        background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                        color: 'white',
                        textDecoration: 'none',
                        borderRadius: '12px',
                        fontWeight: '700',
                        fontSize: '1.05rem',
                        boxShadow: '0 10px 30px rgba(139, 92, 246, 0.4)',
                        transition: 'all 0.3s ease'
                    }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'translateY(-3px)';
                            e.currentTarget.style.boxShadow = '0 15px 40px rgba(139, 92, 246, 0.6)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 10px 30px rgba(139, 92, 246, 0.4)';
                        }}>
                        <i className="ph ph-arrow-right me-2"></i>
                        Get Started Today
                    </a>
                </div>
            </div>

            <style jsx>{`
                .feature-card-modern:hover .card-gradient-overlay {
                    opacity: 1;
                }
            `}</style>
        </section>
    );
}
