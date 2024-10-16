/*
SQLyog Community Edition- MySQL GUI v7.15 
MySQL - 5.5.29 : Database - ewaste
*********************************************************************
*/


/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`ewaste` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `ewaste`;

/*Table structure for table `buser` */

DROP TABLE IF EXISTS `buser`;

CREATE TABLE `buser` (
  `id` int(100) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `location` varchar(100) NOT NULL,
  `ph` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `coins` varchar(10000) DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;


/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
