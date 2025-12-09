'use client';
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export default function MLProcess() {
    const sectionRef = useRef(null);

    useEffect(() => {
        const ctx = gsap.context(() => {
            // Title animation
            gsap.from(".ml-title", {
                scrollTrigger: {
                    trigger: ".ml-process-section",
                    start: "top 80%"
                },
                opacity: 0,
                y: 50,
                duration: 1
            });

            // Timeline animation
            gsap.from(".process-step", {
                scrollTrigger: {
                    trigger: ".ml-process-section",
                    start: "top 70%"
                },
                opacity: 0,
                x: -50,
                stagger: 0.3,
                duration: 0.8,
                ease: "power3.out"
            });
        }, sectionRef);
        return () => ctx.revert();
    }, []);

    const steps = [
        {
            number: "01",
            icon: "📝",
            title: "Input Your Data",
            description: "Share your fitness goals, current stats, dietary preferences, and available equipment. Our AI needs to understand you.",
            color: "#00ccff",
            gradient: "linear-gradient(135deg, #00ccff, #0077ff)"
        },
        {
            number: "02",
            icon: "🤖",
            title: "AI Analysis",
            description: "Our deep learning algorithms process your information, analyzing thousands of data points to understand your unique body type and goals.",
            color: "#10b981",
            gradient: "linear-gradient(135deg, #10b981, #059669)"
        },
        {
            number: "03",
            icon: "⚙️",
            title: "Plan Generation",
            description: "Machine learning models create your personalized 30-day workout and nutrition plan, optimized for your specific requirements.",
            color: "#f59e0b",
            gradient: "linear-gradient(135deg, #f59e0b, #d97706)"
        },
        {
            number: "04",
            icon: "👨‍⚕️",
            title: "Coach Review",
            description: "Professional coaches review your AI-generated plan, adding expert insights and making final adjustments for perfection.",
            color: "#8b5cf6",
            gradient: "linear-gradient(135deg, #8b5cf6, #6d28d9)"
        },
        {
            number: "05",
            icon: "🚀",
            title: "Start Your Journey",
            description: "Receive your complete fitness roadmap and begin your transformation with daily guidance and progress tracking.",
            color: "#ef4444",
            gradient: "linear-gradient(135deg, #ef4444, #dc2626)"
        }
    ];

    return (
        <section className="ml-process-section" ref={sectionRef} style={{
            background: 'linear-gradient(180deg, #0a0a0a 0%, #16213e 50%, #0a0a0a 100%)',
            padding: '100px 0',
            position: 'relative',
            overflow: 'hidden'
        }}>
            {/* Background decorations */}
            <div style={{
                position: 'absolute',
                top: '20%',
                right: '5%',
                width: '500px',
                height: '500px',
                background: 'radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%)',
                filter: 'blur(100px)',
                pointerEvents: 'none'
            }}></div>
            <div style={{
                position: 'absolute',
                bottom: '20%',
                left: '5%',
                width: '400px',
                height: '400px',
                background: 'radial-gradient(circle, rgba(0, 204, 255, 0.1) 0%, transparent 70%)',
                filter: 'blur(100px)',
                pointerEvents: 'none'
            }}></div>

            <div className="container" style={{ position: 'relative', zIndex: 2 }}>
                {/* Section Header */}
                <div style={{ textAlign: 'center', marginBottom: '5rem' }}>
                    <div style={{
                        display: 'inline-block',
                        padding: '8px 20px',
                        background: 'rgba(0, 204, 255, 0.1)',
                        border: '1px solid rgba(0, 204, 255, 0.3)',
                        borderRadius: '50px',
                        color: '#00ccff',
                        fontWeight: '600',
                        fontSize: '0.9rem',
                        marginBottom: '1.5rem'
                    }}>
                        <i className="ph ph-cpu me-2"></i>
                        The Technology Behind Your Transformation
                    </div>
                    <h2 className="ml-title" style={{
                        fontSize: 'clamp(2rem, 5vw, 3.5rem)',
                        fontWeight: '900',
                        marginBottom: '1rem',
                        background: 'linear-gradient(135deg, #fff 0%, #8b5cf6 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text'
                    }}>
                        How Our AI Works
                    </h2>
                    <p style={{
                        fontSize: '1.2rem',
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxWidth: '700px',
                        margin: '0 auto'
                    }}>
                        From data input to your personalized fitness plan in 5 intelligent steps
                    </p>
                </div>

                {/* Process Timeline */}
                <div className="process-timeline" style={{ position: 'relative', maxWidth: '900px', margin: '0 auto' }}>
                    {/* Connecting line */}
                    <div style={{
                        position: 'absolute',
                        left: '40px',
                        top: '60px',
                        bottom: '60px',
                        width: '2px',
                        background: 'linear-gradient(180deg, #00ccff, #8b5cf6, #ef4444)',
                        opacity: 0.3
                    }}></div>

                    {steps.map((step, index) => (
                        <div key={index} className="process-step" style={{
                            position: 'relative',
                            marginBottom: '3rem',
                            paddingLeft: '100px'
                        }}>
                            {/* Step number circle */}
                            <div style={{
                                position: 'absolute',
                                left: '0',
                                top: '0',
                                width: '80px',
                                height: '80px',
                                background: step.gradient,
                                borderRadius: '50%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '2.5rem',
                                boxShadow: `0 10px 40px ${step.color}50`,
                                border: '3px solid rgba(0, 0, 0, 0.5)',
                                zIndex: 2
                            }}>
                                {step.icon}
                            </div>

                            {/* Content card */}
                            <div style={{
                                background: 'rgba(255, 255, 255, 0.03)',
                                backdropFilter: 'blur(20px)',
                                border: `1px solid ${step.color}30`,
                                borderRadius: '16px',
                                padding: '30px',
                                transition: 'all 0.3s ease',
                                cursor: 'pointer',
                                position: 'relative',
                                overflow: 'hidden'
                            }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateX(10px)';
                                    e.currentTarget.style.border = `1px solid ${step.color}60`;
                                    e.currentTarget.style.boxShadow = `0 10px 40px ${step.color}30`;
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateX(0)';
                                    e.currentTarget.style.border = `1px solid ${step.color}30`;
                                    e.currentTarget.style.boxShadow = 'none';
                                }}>
                                {/* Gradient overlay */}
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    background: `linear-gradient(135deg, ${step.color}08 0%, transparent 100%)`,
                                    pointerEvents: 'none'
                                }}></div>

                                {/* Step number badge */}
                                <div style={{
                                    position: 'absolute',
                                    top: '20px',
                                    right: '20px',
                                    fontSize: '3rem',
                                    fontWeight: '900',
                                    color: step.color,
                                    opacity: 0.1
                                }}>
                                    {step.number}
                                </div>

                                {/* Content */}
                                <div style={{ position: 'relative', zIndex: 1 }}>
                                    <h3 style={{
                                        fontSize: '1.6rem',
                                        fontWeight: 'bold',
                                        color: step.color,
                                        marginBottom: '0.75rem'
                                    }}>
                                        {step.title}
                                    </h3>
                                    <p style={{
                                        color: 'rgba(255, 255, 255, 0.8)',
                                        fontSize: '1.05rem',
                                        lineHeight: '1.7',
                                        marginBottom: 0
                                    }}>
                                        {step.description}
                                    </p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Bottom stats/info */}
                <div className="mt-5 pt-5">
                    <div className="row g-4 justify-content-center">
                        <div className="col-md-4 text-center">
                            <div style={{
                                padding: '30px',
                                background: 'rgba(0, 204, 255, 0.05)',
                                backdropFilter: 'blur(10px)',
                                border: '1px solid rgba(0, 204, 255, 0.2)',
                                borderRadius: '16px'
                            }}>
                                <div style={{
                                    fontSize: '3rem',
                                    fontWeight: '900',
                                    background: 'linear-gradient(135deg, #00ccff, #0077ff)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    marginBottom: '0.5rem'
                                }}>
                                    99.8%
                                </div>
                                <div style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '1rem' }}>
                                    AI Accuracy Rate
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4 text-center">
                            <div style={{
                                padding: '30px',
                                background: 'rgba(139, 92, 246, 0.05)',
                                backdropFilter: 'blur(10px)',
                                border: '1px solid rgba(139, 92, 246, 0.2)',
                                borderRadius: '16px'
                            }}>
                                <div style={{
                                    fontSize: '3rem',
                                    fontWeight: '900',
                                    background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    marginBottom: '0.5rem'
                                }}>
                                    2M+
                                </div>
                                <div style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '1rem' }}>
                                    Plans Generated
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4 text-center">
                            <div style={{
                                padding: '30px',
                                background: 'rgba(16, 185, 129, 0.05)',
                                backdropFilter: 'blur(10px)',
                                border: '1px solid rgba(16, 185, 129, 0.2)',
                                borderRadius: '16px'
                            }}>
                                <div style={{
                                    fontSize: '3rem',
                                    fontWeight: '900',
                                    background: 'linear-gradient(135deg, #10b981, #059669)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    marginBottom: '0.5rem'
                                }}>
                                    &lt;3s
                                </div>
                                <div style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '1rem' }}>
                                    Processing Time
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
