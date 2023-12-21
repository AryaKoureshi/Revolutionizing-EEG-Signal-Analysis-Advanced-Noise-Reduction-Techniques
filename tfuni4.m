function [S,A]=tfuni4(e)
% [S,A]=tfuni4(e)
% Comon, version 12 feb 1992
% English comments added in 1994
% Orthogonal real direct transform
% for separating 2 sources in presence of noise
% Sources are assumed zero mean
%
T=length(e);
%%%%% moments d'ordre 2
 g11=e(1,:)*e(1,:)'/T;%cv vers 1
 g22=e(2,:)*e(2,:)'/T;%cv vers 1
 g12=e(1,:)*e(2,:)'/T;%cv vers 0
%%%%% moments d'ordre 4
e2=e.^2;
 g1111=e2(1,:)*e2(1,:)'/T;
 g1112=e2(1,:).*e(1,:)*e(2,:)'/T;
 g1122=e2(1,:)*e2(2,:)'/T;
 g1222=e2(2,:).*e(2,:)*e(1,:)'/T;
 g2222=e2(2,:)*e2(2,:)'/T;
%%%%% cumulants croises d'ordre 4
 q1111=g1111-3*g11*g11;
 q1112=g1112-3*g11*g12;
 q1122=g1122-g11*g22-2*g12*g12;
 q1222=g1222-3*g22*g12;
	q2222=g2222-3*g22*g22;
%%%%% racine de Pw(x): si t est la tangente de l'angle, x=t-1/t.
u=q1111+q2222-6*q1122;v=q1222-q1112;z=q1111*q1111+q2222*q2222;
c4=q1111*q1112-q2222*q1222;
c3=z-4*(q1112*q1112+q1222*q1222)-3*q1122*(q1111+q2222);
c2=3*v*u;
c1=3*z-2*q1111*q2222-32*q1112*q1222-36*q1122*q1122;
c0=-4*(u*v+4*c4);
%c0=q1112*q2222-q1222*q1111-3*q1112*q1111+3*q1222*q2222-6*q1122*q1112+6*q1122*q1222;c0=4*c0
Pw=[c4 c3 c2 c1 c0];R=roots(Pw);I=find(abs(imag(R))<eps);xx=R(I);
%%%%% maximum du contraste en x
a0=q1111;a1=4*q1112;a2=6*q1122;a3=4*q1222;a4=q2222;
b4=a0*a0+a4*a4;
b3=2*(a3*a4-a0*a1);
b2=4*a0*a0+4*a4*a4+a1*a1+a3*a3+2*a0*a2+2*a2*a4;
b1=2*(-3*a0*a1+3*a3*a4+a1*a4+a2*a3-a0*a3-a1*a2);
b0=2*(a0*a0+a1*a1+a2*a2+a3*a3+a4*a4+2*a0*a2+2*a0*a4+2*a1*a3+2*a2*a4);
Pk=[b4 b3 b2 b1 b0];  % numerateur du contraste
Wk=polyval(Pk,xx);Vk=polyval([1 0 8 0 16],xx);Wk=Wk./Vk;
[Wmax,j]=max(Wk);Xsol=xx(j);
%%%%% maximum du contratse en theta
t=roots([1 -Xsol -1]);j=find(t<=1 & t>-1);t=t(j);
%%%%% test et conditionnement
if abs(t)<1/T,
  A=eye(2); %fprintf('pas de rotation plane pour cette paire\n');
else,
  A(1,1)=1/sqrt(1+t*t);A(2,2)=A(1,1);A(1,2)=t*A(1,1);A(2,1)=-A(1,2);
end;
%%%%% filtrage de la sortie
 S=A*e;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
