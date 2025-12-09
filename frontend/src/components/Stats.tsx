'use client';
import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export default function Stats() {
    const sectionRef = useRef(null);
    const [inView, setInView] = useState(false);

    useEffect(() => {
        const ctx = gsap.context(() => {
            ScrollTrigger.create({
                trigger: ".stats-section",
                start: "top 80%",
                onEnter: () => setInView(true)
            });

            gsap.from(".stat-card", {
                scrollTrigger: {
                    trigger: ".stats-section",
                    start: "top 75%",
                    toggleActions: "play none none none"
                },
                scale: 0.9,
                stagger: 0.15,
                duration: 0.6,
                ease: "back.out(1.4)"
            });
        }, sectionRef);
        return () => ctx.revert();
    }, []);

    const stats = [
        {
            value: "50K+",
            label: "Success Stories",
            icon: "🌟",
            color: "#00ccff",
            gradient: "linear-gradient(135deg, #00ccff, #0077ff)"
        },
        {
            value: "2M+",
            label: "Workouts Generated",
            icon: "💪",
            color: "#10b981",
            gradient: "linear-gradient(135deg, #10b981, #059669)"
        },
        {
            value: "98%",
            label: "User Satisfaction",
            icon: "❤️",
            color: "#ef4444",
            gradient: "linear-gradient(135deg, #ef4444, #dc2626)"
        },
        {
            value: "24/7",
            label: "AI Availability",
            icon: "⚡",
            color: "#f59e0b",
            gradient: "linear-gradient(135deg, #f59e0b, #d97706)"
        }
    ];

    return (
        <section className="stats-section" ref={sectionRef} style={{
            background: 'linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%)',
            padding: '100px 0',
            position: 'relative',
            overflow: 'hidden'
        }}>
            <div style={{
                position: 'absolute',
                top: '50%',
                left: '30%',
                transform: 'translate(-50%, -50%)',
                width: '600px',
                height: '600px',
                background: 'radial-gradient(circle, rgba(0, 204, 255, 0.1) 0%, transparent 70%)',
                filter: 'blur(100px)',
                pointerEvents: 'none'
            }}></div>
            <div style={{
                position: 'absolute',
                top: '50%',
                right: '30%',
                transform: 'translate(50%, -50%)',
                width: '500px',
                height: '500px',
                background: 'radial-gradient(circle, rgba(239, 68, 68, 0.1) 0%, transparent 70%)',
                filter: 'blur(100px)',
                pointerEvents: 'none'
            }}></div>

            <div className="container" style={{ position: 'relative', zIndex: 2 }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <div style={{
                        display: 'inline-block',
                        padding: '8px 20px',
                        background: 'rgba(16, 185, 129, 0.2)',
                        border: '1px solid rgba(16, 185, 129, 0.5)',
                        borderRadius: '50px',
                        color: '#34d399',
                        fontWeight: '600',
                        fontSize: '0.9rem',
                        marginBottom: '1.5rem'
                    }}>
                        <i className="ph ph-chart-line-up me-2"></i>
                        Our Impact
                    </div>
                    <h2 style={{
                        fontSize: 'clamp(2rem, 5vw, 3.5rem)',
                        fontWeight: '900',
                        marginBottom: '1rem',
                        color: '#fff',
                        textShadow: '0 0 30px rgba(16, 185, 129, 0.5)'
                    }}>
                        Trusted by Thousands
                    </h2>
                    <p style={{
                        fontSize: '1.2rem',
                        color: '#fff',
                        opacity: 0.8,
                        maxWidth: '700px',
                        margin: '0 auto'
                    }}>
                        Join a thriving community of fitness enthusiasts achieving their goals with AI
                    </p>
                </div>

                <div className="row g-4">
                    {stats.map((stat, index) => (
                        <div className="col-lg-3 col-md-6" key={index}>
                            <div className="stat-card" style={{
                                background: 'rgba(30, 30, 50, 0.95)',
                                backdropFilter: 'blur(20px)',
                                border: `3px solid ${stat.color}`,
                                borderRadius: '20px',
                                padding: '40px 30px',
                                textAlign: 'center',
                                position: 'relative',
                                overflow: 'hidden',
                                transition: 'all 0.3s ease',
                                cursor: 'pointer',
                                height: '100%',
                                boxShadow: `0 8px 32px ${stat.color}50`
                            }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateY(-10px) scale(1.02)';
                                    e.currentTarget.style.boxShadow = `0 20px 60px ${stat.color}70`;
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateY(0) scale(1)';
                                    e.currentTarget.style.boxShadow = `0 8px 32px ${stat.color}50`;
                                }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    background: `radial-gradient(circle at top, ${stat.color}15 0%, transparent 70%)`,
                                    opacity: 0,
                                    transition: 'opacity 0.3s ease',
                                    pointerEvents: 'none'
                                }} className="stat-glow"></div>

                                <div style={{
                                    fontSize: '4rem',
                                    marginBottom: '1.5rem',
                                    filter: `drop-shadow(0 0 20px ${stat.color})`,
                                    position: 'relative',
                                    zIndex: 1
                                }}>
                                    {stat.icon}
                                </div>

                                <div style={{
                                    fontSize: 'clamp(2.5rem, 5vw, 4rem)',
                                    fontWeight: '900',
                                    color: stat.color,
                                    textShadow: `0 0 30px ${stat.color}`,
                                    marginBottom: '0.75rem',
                                    lineHeight: '1',
                                    position: 'relative',
                                    zIndex: 1
                                }}>
                                    {stat.value}
                                </div>

                                <div style={{
                                    color: '#fff',
                                    opacity: 0.9,
                                    fontSize: '1.1rem',
                                    fontWeight: '600',
                                    letterSpacing: '0.5px',
                                    position: 'relative',
                                    zIndex: 1
                                }}>
                                    {stat.label}
                                </div>

                                <div style={{
                                    position: 'absolute',
                                    top: '10px',
                                    right: '10px',
                                    width: '40px',
                                    height: '40px',
                                    borderTop: `2px solid ${stat.color}60`,
                                    borderRight: `2px solid ${stat.color}60`,
                                    borderRadius: '0 16px 0 0',
                                    opacity: 0.5
                                }}></div>
                                <div style={{
                                    position: 'absolute',
                                    bottom: '10px',
                                    left: '10px',
                                    width: '40px',
                                    height: '40px',
                                    borderBottom: `2px solid ${stat.color}60`,
                                    borderLeft: `2px solid ${stat.color}60`,
                                    borderRadius: '0 0 0 16px',
                                    opacity: 0.5
                                }}></div>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-5 pt-5">
                    <div style={{
                        maxWidth: '800px',
                        margin: '0 auto',
                        background: 'rgba(30, 30, 50, 0.9)',
                        backdropFilter: 'blur(20px)',
                        border: '2px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '20px',
                        padding: '40px',
                        textAlign: 'center'
                    }}>
                        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>💬</div>
                        <p style={{
                            fontSize: '1.3rem',
                            fontStyle: 'italic',
                            color: '#fff',
                            opacity: 0.95,
                            lineHeight: '1.8',
                            marginBottom: '1.5rem'
                        }}>
                            "The AI-generated plan was incredibly accurate. I lost 15 pounds in 30 days and gained so much muscle definition. This is the future of fitness!"
                        </p>
                        <div style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '12px'
                        }}>
                            <div style={{
                                width: '50px',
                                height: '50px',
                                borderRadius: '50%',
                                background: 'linear-gradient(135deg, #00ccff, #0077ff)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '1.5rem',
                                fontWeight: 'bold',
                                color: 'white'
                            }}>
                                JD
                            </div>
                            <div style={{ textAlign: 'left' }}>
                                <div style={{ color: 'white', fontWeight: 'bold', fontSize: '1.1rem' }}>
                                    John Doe
                                </div>
                                <div style={{ color: '#fff', opacity: 0.7, fontSize: '0.9rem' }}>
                                    Fitness Enthusiast
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <style jsx>{`
                .stat-card:hover .stat-glow {
                    opacity: 1;
                }
            `}</style>
        </section>
    );
}
